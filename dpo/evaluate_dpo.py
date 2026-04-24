"""
evaluate_dpo.py
===============
LLM-as-a-judge evaluation for DPO-trained medAlpaca models.

Compares two models (e.g. SFT baseline vs DPO fine-tune) using a FREE
HuggingFace-hosted judge model — no paid API required.

The judge runs via the HuggingFace Inference API (serverless, free tier).
Recommended free judge models:
  - meta-llama/Llama-3.1-8B-Instruct   (best quality, needs HF access request)
  - HuggingFaceH4/zephyr-7b-beta        (good, no access request needed)

Get a free HuggingFace token at: https://huggingface.co/settings/tokens
Set it via:  export HF_TOKEN=hf_...   or pass --hf_token hf_...

Scoring modes (--mode):
  pairwise   : Judge picks which of two responses (A vs B) is better.
               Reports win/tie/loss rate. Best for comparing SFT vs DPO.
  absolute   : Judge rates a single response 1-5 on multiple criteria.
               Useful when you don't have a baseline to compare against.
  both       : Runs pairwise then absolute.

Output:
  - JSON file with all scores and judge reasoning
  - Printed summary table

Usage:
    # Pairwise: compare SFT vs DPO (Mistral judge, no access request needed)
    python evaluate_dpo.py \\
        --mode pairwise \\
        --model_a ./sft_model \\
        --model_b ./output/dpo \\
        --judge_model mistralai/Mistral-7B-Instruct-v0.3 \\
        --input_path mediqa_eval_ready.json \\
        --output_path eval_results.json \\
        --hf_token hf_...

    # Pairwise with Llama-3.1-8B judge
    python evaluate_dpo.py \\
        --mode pairwise \\
        --model_a ./sft_model \\
        --model_b ./output/dpo \\
        --judge_model meta-llama/Llama-3.1-8B-Instruct \\
        --input_path mediqa_eval_ready.json \\
        --output_path eval_results.json

    # Input file: JSON array with at least "instruction" field.
    # Optionally pre-computed: "response_a" / "response_b" fields
    # (if present, generation for that model is skipped).
"""

import json
import os
import re
import sys
import time
from pathlib import Path
import requests

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_checkpoint(checkpoint_path: str) -> dict:
    p = Path(checkpoint_path)
    if p.exists():
        with open(p, 'r') as f:
            data = json.load(f)
        print(f"Resuming from checkpoint: {len(data['results'])} pairs already evaluated.")
        return data
    return {
        "results": [], "wins_a": 0, "wins_b": 0, "ties": 0, "errors": 0,
        "scores_a_total": {"pers": []},
        "scores_b_total": {"pers": []}
    }

def save_checkpoint(checkpoint_path: str, state: dict):
    with open(checkpoint_path, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ALPACA_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
ALPACA_PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

PAIRWISE_JUDGE_PROMPT = """You are a senior medical communication expert comparing two AI responses.

## Question
{instruction}{input_block}

## Response A
{response_a}

## Response B
{response_b}

## Evaluation Rubric
Rate both responses on a scale of 1-5 for Persona Alignment only:

- If instruction says 'I am a doctor': Does it use abundant medical terminology, technical jargon, and clinical language as found in a medical textbook? 
  - 1 = uses plain language, no medical terms
  - 3 = some medical terms but not consistently technical
  - 5 = rich medical terminology, highly clinical and textbook-like
  
- If instruction says 'I am a patient': Is it simple, easy to understand, and free of unnecessary jargon?
  - 1 = full of medical jargon, hard to understand
  - 3 = somewhat simple but still contains jargon
  - 5 = fully plain language, easy for a non-medical person to understand

## Instructions
- Evaluate Response A and Response B independently.
- Focus ONLY on Persona Alignment, ignore other aspects.
- TIES ARE DISCOURAGED.
- Provide a brief justification.

### Scorecard
- Response A: [Pers: X]
- Response B: [Pers: X]

### Comparison Reasoning:
[Provide 1-2 sentences focused on persona alignment only]

### Final Verdict:
VERDICT: <A, B, or TIE>"""


# ---------------------------------------------------------------------------
# Free HuggingFace Inference API judge
# ---------------------------------------------------------------------------
#
# Recommended free judge models (pass as --judge_model):
#   mistralai/Mistral-7B-Instruct-v0.3   ← no access request needed
#   HuggingFaceH4/zephyr-7b-beta         ← no access request needed
#   meta-llama/Llama-3.1-8B-Instruct     ← requires HF access request
#
# Get a free token at: https://huggingface.co/settings/tokens

import requests
import json
import time

def call_hf_judge(
    prompt: str,
    hf_token: str,
    judge_model: str = "Qwen/Qwen2.5-7B-Instruct", # Most stable for 2026 Router
    max_new_tokens: int = 512,
    retries: int = 5,
    retry_delay: float = 10.0,
) -> str:
    # THE 2026 UNIFIED ROUTER URL
    # This automatically picks the best provider (HF, Together, Sambanova, etc.)
    url = "https://router.huggingface.co/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    # Some models (e.g. Mistral) don't support system role
    # Merge system message into user message instead
    if "mistral" in judge_model.lower():
        messages = [{"role": "user", "content": "You are a helpful assistant and a neutral judge.\n\n" + prompt}]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant and a neutral judge."},
            {"role": "user", "content": prompt}
        ]

    payload = {
        "model": judge_model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.01
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            # If model is loading (503) or rate limited (429)
            if response.status_code in [503, 429]:
                print(f"  Provider busy/loading ({response.status_code}), waiting {retry_delay * attempt}s...")
                time.sleep(retry_delay * attempt)
                continue
            
            # If the model simply isn't available on the router at all
            if response.status_code == 404:
                print(f"Model {judge_model} not found on any Router provider.")
                return "ERROR: MODEL_NOT_AVAILABLE"

            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()

        except Exception as e:
            if attempt == retries:
                print(f"❌ API Error: {e}")
                raise e
            time.sleep(retry_delay)
            
    return "ERROR: JUDGE_TIMEOUT"




# ---------------------------------------------------------------------------
# Local model helpers  (generation only — no longer used for judging)
# ---------------------------------------------------------------------------

def build_prompt(instruction: str, input_text: str = "") -> str:
    if input_text and input_text.strip():
        return ALPACA_PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    return ALPACA_PROMPT_WITHOUT_INPUT.format(instruction=instruction)


def load_model(path: str, load_in_4bit: bool = False, load_in_8bit: bool = False):
    path = str(Path(path).resolve())
    is_lora = (Path(path) / "adapter_config.json").exists()

    bnb = None
    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        bnb = BitsAndBytesConfig(load_in_8bit=True)

    dtype = torch.float16 if not (load_in_4bit or load_in_8bit) else None

    if is_lora:
        from peft import PeftModel
        from accelerate import dispatch_model, infer_auto_device_map

        cfg = json.loads((Path(path) / "adapter_config.json").read_text())
        base = cfg.get("base_model_name_or_path", "")
        if not base:
            sys.exit(f"Cannot infer base model from {path}/adapter_config.json")
        base = str(Path(base).resolve()) if base.startswith(".") else base

        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=bnb, torch_dtype=dtype,
            device_map=None, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, path, is_trainable=False)
        model = model.merge_and_unload()

        # Now it is a plain nn.Module — safe to dispatch across GPUs/CPU.
        # Pass class names as plain strings; avoids the set-hashing bug in
        # accelerate <= 0.29 where sets were passed instead of lists.
        no_split = [
            "LlamaDecoderLayer", "MistralDecoderLayer",
            "Qwen2DecoderLayer", "FalconDecoderLayer",
            "GPTNeoXLayer", "BloomBlock",
        ]
        device_map = infer_auto_device_map(
            model, no_split_module_classes=no_split,
        )
        model = dispatch_model(model, device_map=device_map)

        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb, torch_dtype=dtype,
            device_map="auto", trust_remote_code=True,
        )
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()
    return model, tok


def generate_batch(
    model, tokenizer, prompts: list,
    max_new_tokens: int = 200,   # 200 is plenty for eval; 512 was the bottleneck
    temperature: float = 0.1,
    top_p: float = 0.75,
) -> list:
    """Generate responses for a batch of prompts in a single forward pass."""
    enc = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=1024,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens for each item in the batch
    return [
        tokenizer.decode(out[i, input_len:], skip_special_tokens=True).strip()
        for i in range(len(prompts))
    ]


def generate_response(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.1,
    top_p: float = 0.75,
) -> str:
    """Single-prompt wrapper around generate_batch (used by run_absolute)."""
    return generate_batch(model, tokenizer, [prompt],
                          max_new_tokens, temperature, top_p)[0]


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------

def unload_model(model) -> None:
    """Move model off GPU and free VRAM before loading the next model."""
    try:
        model.cpu()
    except Exception:
        pass
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Parsing judge output
# ---------------------------------------------------------------------------

def parse_pairwise(text: str) -> str:
    """Returns 'A', 'B', or 'TIE'."""
    match = re.search(r"VERDICT\s*:\s*(A|B|TIE)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for token in reversed(text.split()):
        t = token.strip("*.,\n").upper()
        if t in ("A", "B", "TIE"):
            return t
    return "PARSE_ERROR"


def parse_absolute(text: str) -> dict:
    """Returns dict of scores."""
    scores = {}
    for key in ("ACCURACY", "COMPLETENESS", "CLARITY", "SAFETY", "OVERALL"):
        m = re.search(rf"{key}\s*:\s*([1-5])", text, re.IGNORECASE)
        scores[key.lower()] = int(m.group(1)) if m else None
    m = re.search(r"REASONING\s*:\s*(.+)", text, re.IGNORECASE)
    scores["reasoning"] = m.group(1).strip() if m else text[-200:]
    return scores


# ---------------------------------------------------------------------------
# Pairwise evaluation
# ---------------------------------------------------------------------------

def generate_all_responses(
    model, tokenizer, records: list, label: str,
    batch_size: int = 8,
    max_new_tokens: int = 200,
) -> list:
    """
    Generate responses for all records using batched inference.
    Batching amortises the per-call overhead and keeps the GPU saturated,
    cutting total generation time by ~batch_size× vs one-at-a-time.
    """
    # Separate pre-cached from records that need generation
    responses = [None] * len(records)
    pending_indices, pending_prompts = [], []

    for i, r in enumerate(records):
        cached = r.get(f"response_{label.lower()}")
        if cached:
            responses[i] = cached
        else:
            instruction = r.get("instruction", r.get("question", ""))
            input_text  = r.get("input", "")
            pending_indices.append(i)
            pending_prompts.append(build_prompt(instruction, input_text))

    total = len(pending_prompts)
    generated = 0
    for batch_start in range(0, total, batch_size):
        batch_prompts  = pending_prompts[batch_start: batch_start + batch_size]
        batch_indices  = pending_indices[batch_start: batch_start + batch_size]
        batch_responses = generate_batch(
            model, tokenizer, batch_prompts, max_new_tokens=max_new_tokens,
        )
        for idx, resp in zip(batch_indices, batch_responses):
            responses[idx] = resp
        generated += len(batch_prompts)
        print(f"  [Model {label}] {generated}/{total} responses generated…")

    return responses

def save_generated_responses(records, responses_a, responses_b, filename="generated_outputs.json"):
    combined_data = []
    for i, r in enumerate(records):
        # Create a new record that includes the original data + the new responses
        new_entry = r.copy()
        new_entry["response_a"] = responses_a[i]
        new_entry["response_b"] = responses_b[i]
        combined_data.append(new_entry)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    print(f"Generation complete. Saved {len(combined_data)} responses to {filename}")


def run_pairwise(
    records,
    model_a_path: str, model_b_path: str,
    hf_token: str,
    judge_model: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    swap_positions: bool = True,
    batch_size: int = 8,
    max_new_tokens: int = 200,
    checkpoint_path: str = "./eval_set/checkpoint.json",  # ← add this
):
    """
    Runs pairwise evaluation and extracts rubric scores (1-5) 
    for Accuracy, Completeness, Clarity, and Safety.
    """
    # ── Phase 0: Check for precomputed responses ──────────────────────────────
    first_record = records[0]
    has_precomputed = first_record.get("response_a") and first_record.get("response_b")

    if has_precomputed:
        print("Pre-computed responses found. Skipping local model loading!")
        responses_a = [r.get("response_a") for r in records]
        responses_b = [r.get("response_b") for r in records]
    else:
        # Generate A
        print(f"Loading model A: {model_a_path}")
        m_a, t_a = load_model(model_a_path, load_in_4bit, load_in_8bit)
        responses_a = generate_all_responses(m_a, t_a, records, "A", batch_size, max_new_tokens)
        unload_model(m_a)
        
        # Generate B
        print(f"Loading model B: {model_b_path}")
        m_b, t_b = load_model(model_b_path, load_in_4bit, load_in_8bit)
        responses_b = generate_all_responses(m_b, t_b, records, "B", batch_size, max_new_tokens)
        unload_model(m_b)
        save_generated_responses(records, responses_a, responses_b, "./eval_set/responses.json")

    # ── Phase 3: Judging and Rubric Extraction ────────────────────────────────
    print(f"\nJudging {len(records)} pairs via {judge_model}...")
    state = load_checkpoint(checkpoint_path)
    already_done      = len(state["results"])
    wins_a            = state["wins_a"]
    wins_b            = state["wins_b"]
    ties              = state["ties"]
    errors            = state["errors"]
    scores_a_total    = state["scores_a_total"]
    scores_b_total    = state["scores_b_total"]
    results           = state["results"]

    print(f"\nJudging {len(records)} pairs via {judge_model}...")
    if already_done > 0:
        print(f"Skipping first {already_done} pairs (already in checkpoint).")

    for i, (r, resp_a, resp_b) in enumerate(zip(records, responses_a, responses_b), 1):
        if i <= already_done:
            continue  # ← skip already evaluated pairs

        instruction = r.get("instruction", r.get("question", ""))
        input_text  = r.get("input", "")
        input_block = f"\n\n**Context:** {input_text}" if input_text.strip() else ""

        verdicts  = []
        raw_outputs = []  # ← store raw judge outputs for this pair
        orderings = [(resp_a, resp_b, "A", "B")]
        if swap_positions:
            orderings.append((resp_b, resp_a, "B", "A"))

        for idx, (r1, r2, l1, l2) in enumerate(orderings):
            judge_prompt = PAIRWISE_JUDGE_PROMPT.format(
                instruction=instruction, input_block=input_block,
                response_a=r1, response_b=r2
            )
            raw = call_hf_judge(judge_prompt, hf_token=hf_token, judge_model=judge_model)

            print(f"\n--- Judge Raw Output (item {i}, pass {idx+1}) ---")
            print(raw)
            print("---------------------------------------------------\n")

            raw_outputs.append(raw)  # ← save raw output

            v = parse_pairwise(raw)
            verdicts.append(l1 if v == "A" else l2 if v == "B" else v)

            if idx == 0:
                for line in raw.splitlines():
                    line = re.sub(r'\*+', '', line)
                    line = line.strip('- ').strip()
                    # Match: - Response A: [Pers: X]
                    m = re.match(r"Response\s+([AB])\s*:\s*\[Pers:\s*([1-5])\]", line, re.IGNORECASE)
                    if m:
                        label = m.group(1).upper()
                        val = int(m.group(2))
                        target = scores_a_total if label == "A" else scores_b_total
                        target["pers"].append(val)

        final = verdicts[0] if len(set(verdicts)) == 1 else "TIE"
        if final == "A":     wins_a += 1
        elif final == "B":   wins_b += 1
        elif final == "TIE": ties   += 1
        else:                errors += 1

        results.append({
            "instruction": instruction,
            "response_a":  resp_a,
            "response_b":  resp_b,
            "verdicts":    verdicts,
            "final":       final,
            "judge_raw":   raw_outputs,  # ← full raw outputs saved here
        })

        # ── Save checkpoint after every pair ──────────────────────────────────
        state.update({
            "results": results,
            "wins_a": wins_a, "wins_b": wins_b,
            "ties": ties, "errors": errors,
            "scores_a_total": scores_a_total,
            "scores_b_total": scores_b_total,
        })
        save_checkpoint(checkpoint_path, state)

        print(f"  [{i}/{len(records)}] {final} | A:{wins_a} B:{wins_b} TIE:{ties}")
        time.sleep(3)

    total = len(records)
    summary = {
        "total": total,
        "model_a_wins": wins_a, "model_b_wins": wins_b, "ties": ties,
        "model_a_win_%": round(100 * wins_a / total, 1),
        "model_b_win_%": round(100 * wins_b / total, 1),
        "tie_%":         round(100 * ties   / total, 1),
        "scores_a": {m: round(sum(v)/len(v), 2) if v else 0 for m, v in scores_a_total.items()},
        "scores_b": {m: round(sum(v)/len(v), 2) if v else 0 for m, v in scores_b_total.items()},
    }
    return results, summary

# ---------------------------------------------------------------------------
# Absolute evaluation
# ---------------------------------------------------------------------------

def run_absolute(records, model_a_path: str, hf_token: str, judge_model: str,
                  load_in_4bit: bool = False, load_in_8bit: bool = False):
    print(f"Loading model A: {model_a_path}")
    model_a, tok_a = load_model(model_a_path, load_in_4bit, load_in_8bit)
    results = []
    totals  = {"accuracy": 0, "completeness": 0, "clarity": 0, "safety": 0, "overall": 0}
    n_valid = 0

    for i, r in enumerate(records, 1):
        instruction = r.get("instruction", r.get("question", ""))
        input_text  = r.get("input", "")
        input_block = f"\n\n**Context:** {input_text}" if input_text.strip() else ""

        response = r.get("response_a") or generate_response(
            model_a, tok_a, build_prompt(instruction, input_text))

        judge_prompt = ABSOLUTE_JUDGE_PROMPT.format(
            instruction=instruction,
            input_block=input_block,
            response=response,
        )
        raw = call_hf_judge(judge_prompt, hf_token=hf_token, judge_model=judge_model)
        scores = parse_absolute(raw)

        results.append({
            "instruction": instruction,
            "input":       input_text,
            "response":    response,
            "scores":      scores,
            "judge_raw":   raw,
        })

        if all(scores[k] is not None for k in totals):
            for k in totals:
                totals[k] += scores[k]
            n_valid += 1

        print(f"  [{i}/{len(records)}] overall={scores.get('overall')}  "
              f"acc={scores.get('accuracy')}  "
              f"complete={scores.get('completeness')}  "
              f"clarity={scores.get('clarity')}  "
              f"safety={scores.get('safety')}")

    summary = {k: round(v / n_valid, 2) for k, v in totals.items()} if n_valid else {}
    summary["n_scored"] = n_valid
    summary["n_total"]  = len(records)
    return results, summary


# ---------------------------------------------------------------------------
# Pretty-print summaries
# ---------------------------------------------------------------------------

def print_pairwise_summary(summary: dict, name_a: str, name_b: str, judge_model: str):
    print("\n" + "=" * 65)
    print(f"  PAIRWISE EVALUATION SUMMARY  (judge: {judge_model})")
    print("=" * 65)
    print(f"  {name_a:<22} wins: {summary['model_a_wins']:>4}  ({summary['model_a_win_%']}%)")
    print(f"  {name_b:<22} wins: {summary['model_b_wins']:>4}  ({summary['model_b_win_%']}%)")
    print(f"  {'Ties':<22}     : {summary['ties']:>4}  ({summary['tie_%']}%)")

    print("-" * 65)
    print(f"{'Metric':<20} | {name_a[:18]:<20} | {name_b[:18]:<20}")
    print("-" * 65)

    metric_labels = {
        "acc":  "Accuracy",
        "pers": "Persona Alignment",
        "clar": "Clarity",
        "safe": "Safety",
    }

    for m, label in metric_labels.items():
        val_a = summary["scores_a"].get(m, 0)
        val_b = summary["scores_b"].get(m, 0)
        mark_a = "★" if val_a > val_b else " "
        mark_b = "★" if val_b > val_a else " "
        print(f"{label:<20} | {val_a:<18.2f} {mark_a} | {val_b:<18.2f} {mark_b}")  # ← inside loop

    print("=" * 65 + "\n")


def print_absolute_summary(summary: dict, name: str, judge_model: str):
    print("\n" + "=" * 55)
    print(f"  ABSOLUTE EVALUATION SUMMARY  (judge: {judge_model})")
    print(f"  Model: {name}")
    print("=" * 55)
    for k in ("accuracy", "completeness", "clarity", "safety", "overall"):
        if k in summary:
            print(f"  {k.capitalize():<16}: {summary[k]:.2f} / 5.00")
    print(f"  Scored {summary.get('n_scored')}/{summary.get('n_total')} questions")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    # ── Models to evaluate ────────────────────────────────────────────────────
    model_a: str = "",              # SFT baseline (or only model for absolute)
    model_b: str = "",              # DPO model    (pairwise only)
    # ── Free HuggingFace judge ────────────────────────────────────────────────
    judge_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    #   Other good free options:
    #     "HuggingFaceH4/zephyr-7b-beta"
    #     "meta-llama/Llama-3.1-8B-Instruct"  (requires HF access request)
    hf_token: str = "",             # or set HF_TOKEN env var
    # ── Evaluation mode ───────────────────────────────────────────────────────
    mode: str = "pairwise",        # "pairwise" | "absolute" | "both"
    # ── Data ─────────────────────────────────────────────────────────────────
    input_path: str = "mediqa_eval_ready.json",
    checkpoint_path: str = "./eval_set/checkpoint.json", 
    output_path: str = "eval_results.json",
    max_samples: int = 50,
    # ── Quantisation (applied to both local generation models) ────────────────
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    # ── Generation ───────────────────────────────────────────────────────────
    batch_size: int = 8,           # prompts per forward pass (increase if VRAM allows)
    max_new_tokens: int = 200,     # cap response length; 200 is enough for eval
    # ── Pairwise option ───────────────────────────────────────────────────────
    swap_positions: bool = True,   # run each pair twice to reduce positional bias
    # ── Display labels ────────────────────────────────────────────────────────
    name_a: str = "Model-A (SFT)",
    name_b: str = "Model-B (DPO)",
):
    # ── Resolve HF token ──────────────────────────────────────────────────────
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        sys.exit(
            "HuggingFace token required.\n"
            "  Pass --hf_token hf_...  or  export HF_TOKEN=hf_...\n"
            "  Get a free token at: https://huggingface.co/settings/tokens"
        )

    # ── Validate args ─────────────────────────────────────────────────────────
    if not model_a:
        sys.exit("--model_a is required.")
    if mode in ("pairwise", "both") and not model_b:
        sys.exit("--model_b is required for pairwise / both mode.")

    # ── Load data ─────────────────────────────────────────────────────────────
    raw     = Path(input_path).read_text(encoding="utf-8").strip()
    records = json.loads(raw) if raw.startswith("[") else \
              [json.loads(l) for l in raw.splitlines() if l.strip()]
    if max_samples > 0:
        records = records[:max_samples]
    print(f"Evaluating {len(records)} questions  (mode={mode}, judge={judge_model})\n")

    all_output = {}

    # ── Pairwise: load A → generate → unload → load B → generate → unload → judge
    if mode in ("pairwise", "both"):
        print(f"\n── Pairwise evaluation (judge: {judge_model}) ──")
        pair_results, pair_summary = run_pairwise(
            records,
            model_a_path=model_a, model_b_path=model_b,
            hf_token=token, judge_model=judge_model,
            load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
            swap_positions=swap_positions,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            checkpoint_path=checkpoint_path,
        )
        all_output["pairwise_results"] = pair_results
        all_output["pairwise_summary"] = pair_summary
        print_pairwise_summary(pair_summary, name_a, name_b, judge_model)

    # ── Absolute: load A → generate → unload → judge ──────────────────────────
    if mode in ("absolute", "both"):
        print(f"\n── Absolute evaluation (judge: {judge_model}) ──")
        abs_results, abs_summary = run_absolute(
            records,
            model_a_path=model_a,
            hf_token=token, judge_model=judge_model,
            load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
        )
        all_output["absolute_results"] = abs_results
        all_output["absolute_summary"] = abs_summary
        print_absolute_summary(abs_summary, name_a, judge_model)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_path).write_text(
        json.dumps(all_output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Full results saved → {output_path}")


if __name__ == "__main__":
    fire.Fire(main)