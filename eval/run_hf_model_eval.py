#!/usr/bin/env python3
"""
Evaluate a Hugging Face causal LM on MediQA val set or PubMedQA (long_answer) for generalization.
Outputs BERTScore, BLEU, ROUGE and saves to results/eval_results_{model_name}_{dataset}.json.
Supports --dataset mediqa (default) or pubmedqa; PubMedQA uses a subset (default 200 samples).

Usage summary: test different HF models on MediQA (10% val) and PubMedQA (200 samples).

  # MediQA: same 10% val split as training (no --max_eval_samples = use all val samples)
  python eval/run_hf_model_eval.py --model_id <hf_model_id> [--load_in_4bit]
  python eval/run_hf_model_eval.py --model_id meta-llama/Llama-3.1-8B-Instruct --load_in_4bit
  python eval/run_hf_model_eval.py --model_id medical_lora_output/final_model --peft

  # PubMedQA: 200 samples for generalization (long_answer as reference)
  python eval/run_hf_model_eval.py --model_id <hf_model_id> --dataset pubmedqa [--load_in_4bit]
  python eval/run_hf_model_eval.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset pubmedqa --load_in_4bit
  python eval/run_hf_model_eval.py --model_id medical_lora_output/final_model --dataset pubmedqa --peft

  # Optional: --max_eval_samples N to cap MediQA; PubMedQA default is 200
  # Optional: --use_cache to skip generation and only recompute metrics
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

EVAL_DIR = os.path.join(ROOT, "eval")
RESULTS_DIR = os.path.join(ROOT, "results")
MEDIQA_NAME = "medalpaca/medical_meadow_mediqa"
PUBMEDQA_NAME = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
PUBMEDQA_DEFAULT_MAX = 200  # subset for PubMedQA (pqa_labeled has ~1k)
VAL_SPLIT = 0.1
SEED = 42


def sanitize_model_name(model_id):
    """Turn model_id into a safe filename segment (e.g. meta-llama/Llama-3.1-8B -> meta-llama__Llama-3.1-8B)."""
    s = model_id.strip().replace("/", "__")
    s = re.sub(r"[^\w\-.]", "_", s)
    return s[:120]  # cap length


def load_eval_dataset(dataset_kind="mediqa", token=None, max_samples=None):
    """
    Load eval data. dataset_kind: "mediqa" | "pubmedqa".
    mediqa: same validation split as training (seed=42, 10%).
    pubmedqa: qiaojin/PubMedQA pqa_labeled, long_answer as reference; subset capped by max_samples (default PUBMEDQA_DEFAULT_MAX).
    """
    from datasets import load_dataset
    if dataset_kind == "mediqa":
        dataset = load_dataset(MEDIQA_NAME, split="train", token=token)
        split = dataset.train_test_split(test_size=VAL_SPLIT, seed=SEED)
        out = split["test"]
        if max_samples is not None and max_samples < len(out):
            out = out.select(range(max_samples))
        return out, MEDIQA_NAME
    if dataset_kind == "pubmedqa":
        dataset = load_dataset(PUBMEDQA_NAME, PUBMEDQA_CONFIG, split="train", token=token)
        n = len(dataset)
        cap = max_samples if max_samples is not None else PUBMEDQA_DEFAULT_MAX
        if cap < n:
            dataset = dataset.select(range(cap))
        return dataset, f"{PUBMEDQA_NAME}({PUBMEDQA_CONFIG})"
    raise ValueError(f"Unknown dataset_kind: {dataset_kind}")


def generate_answer(model, tokenizer, question, context=None, max_new_tokens=256, temperature=0.1, top_p=0.95):
    """Same Llama 3 chat format and decode logic as eval/inference.py."""
    import torch
    if context:
        question_text = f"{question}\nContext: {context}"
    else:
        question_text = question
    prompt = f"""<|start_header_id|>user<|end_header_id|>
{question_text}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    try:
        eot_token_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        eot_token_id = eot_token_id[0] if eot_token_id else None
    except Exception:
        eot_token_id = None
    eos_token_id = tokenizer.eos_token_id
    stop_token_ids = []
    if eot_token_id is not None and eot_token_id != eos_token_id:
        stop_token_ids.append(eot_token_id)
    if eos_token_id is not None:
        stop_token_ids.append(eos_token_id)
    final_eos_token_id = stop_token_ids[0] if stop_token_ids else tokenizer.eos_token_id
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=final_eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    generated_tokens = outputs[0][input_length:]
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    answer = answer.replace("<|eot_id|>", "").strip().replace("<|end_of_text|>", "").strip()
    if "<|start_header_id|>" in answer:
        answer = answer.split("<|start_header_id|>")[0].strip()
    if not answer or len(answer.strip()) < 5:
        marker = "<|start_header_id|>assistant<|end_header_id|>\n"
        if marker in full_response:
            answer = full_response.split(marker)[-1].strip()
            answer = answer.replace("<|eot_id|>", "").strip()
            if "<|start_header_id|>" in answer:
                answer = answer.split("<|start_header_id|>")[0].strip()
    return answer


def run_bertscore(candidates, references, device=None):
    """Compute BERTScore (P, R, F1)."""
    from bert_score import score
    P, R, F1 = score(candidates, references, lang="en", device=device, verbose=True)
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def run_bleu(candidates, references):
    """Compute corpus BLEU (sacrebleu). ref_streams = [references] for one ref per segment."""
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(candidates, [references])
    return {"bleu": bleu.score}


def run_rouge(candidates, references):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L (F1)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for c, r in zip(candidates, references):
        s = scorer.score(r, c)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    import numpy as np
    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rl)),
    }


def main():
    import argparse
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig

    p = argparse.ArgumentParser(description="Evaluate a Hugging Face model on medical_meadow_mediqa val set; output BERTScore, BLEU, ROUGE")
    p.add_argument("--model_id", type=str, required=True, help="Hugging Face model id (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    p.add_argument("--dataset", type=str, default="mediqa", choices=["mediqa", "pubmedqa"], help="Eval dataset: mediqa (default) or pubmedqa (long_answer, subset)")
    p.add_argument("--peft", action="store_true", help="Load as PEFT adapter (AutoPeftModelForCausalLM)")
    p.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (saves memory)")
    p.add_argument("--max_eval_samples", type=int, default=None, help="Cap samples (default: all for mediqa, %d for pubmedqa)" % PUBMEDQA_DEFAULT_MAX)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--use_cache", action="store_true", help="Load predictions from cache, only recompute metrics")
    args = p.parse_args()

    model_id = args.model_id.strip()
    dataset_kind = args.dataset.strip().lower()
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token and ("meta-llama" in model_id or "llama" in model_id.lower()):
        print("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN for gated models.")
        sys.exit(1)

    name_slug = sanitize_model_name(model_id)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache_path = os.path.join(EVAL_DIR, f".cache_hf_{name_slug}_{dataset_kind}.json")
    out_path = os.path.join(RESULTS_DIR, f"eval_results_{name_slug}_{dataset_kind}.json")

    if args.use_cache and os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        candidates = [str(c).strip() if c is not None else "" for c in data["candidates"]]
        references = [str(r).strip() if r is not None else "" for r in data["references"]]
        dataset_label = data.get("dataset", dataset_kind)
        print(f"Loaded {len(candidates)} predictions from cache: {cache_path}")
    else:
        print(f"Loading model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if args.peft:
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                token=hf_token,
            )
        else:
            if args.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="cuda:0",
                    trust_remote_code=True,
                    token=hf_token,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="cuda:0",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    token=hf_token,
                )
        print("Loading evaluation dataset...")
        eval_dataset, dataset_label = load_eval_dataset(dataset_kind, token=hf_token, max_samples=args.max_eval_samples)
        n_total = len(eval_dataset)
        print(f"Dataset: {dataset_label}, evaluating on {n_total} samples")
        candidates = []
        references = []
        for i, row in enumerate(eval_dataset):
            if dataset_kind == "mediqa":
                question = row.get("instruction", "") or ""
                context = row.get("input", "") or ""
                ref = row.get("output", "") or ""
            else:
                # pubmedqa: qiaojin/PubMedQA uses question, context (list or dict with "contexts"), long_answer
                question = row.get("question", "") or ""
                ctx = row.get("context", "")
                if isinstance(ctx, dict) and "contexts" in ctx:
                    context = " ".join(ctx["contexts"]) if ctx["contexts"] else ""
                elif isinstance(ctx, list):
                    context = " ".join(str(t) for t in ctx)
                else:
                    context = str(ctx) if ctx else ""
                ref = row.get("long_answer", "") or ""
            question = str(question).strip()
            context = str(context).strip() if context else ""
            ref = str(ref).strip()
            pred = generate_answer(
                model, tokenizer, question, context=context or None,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=0.95,
            )
            candidates.append(pred)
            references.append(ref)
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{n_total}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"model_id": model_id, "dataset": dataset_label, "candidates": candidates, "references": references}, f, ensure_ascii=False)
        print(f"Saved predictions to {cache_path}")

    device = "cuda:0" if torch.cuda.is_available() else None
    metrics = {}

    print("Computing BERTScore...")
    metrics.update(run_bertscore(candidates, references, device=device))

    print("Computing BLEU...")
    metrics.update(run_bleu(candidates, references))

    print("Computing ROUGE...")
    metrics.update(run_rouge(candidates, references))

    payload = {
        "model_id": model_id,
        "dataset": dataset_label,
        "dataset_kind": dataset_kind,
        "n_samples": len(candidates),
        "metrics": metrics,
    }
    if dataset_kind == "mediqa":
        payload["val_split"] = VAL_SPLIT
        payload["seed"] = SEED
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Results for {model_id}")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
    print("=" * 50)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
