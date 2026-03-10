#!/usr/bin/env python3
"""
Evaluate fine-tuned medical Q&A model with BERTScore.
Uses the same validation split as training (medical_meadow_mediqa, 10%, seed=42).
"""

import os
from datasets import load_dataset
from dotenv import load_dotenv

# Project root (parent of eval/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(ROOT, "medical_lora_output", "final_model")
load_dotenv(os.path.join(ROOT, ".env"))


def load_eval_dataset(dataset_name="medalpaca/medical_meadow_mediqa", val_split=0.1, seed=42, token=None):
    """Load dataset and get validation split (same as finetune_lora.py)."""
    dataset = load_dataset(dataset_name, split="train", token=token)
    if val_split <= 0:
        return dataset
    split = dataset.train_test_split(test_size=val_split, seed=seed)
    return split["test"]


def run_bertscore(candidates, references, model_type=None, lang="en", device=None):
    """Compute BERTScore (P, R, F1)."""
    from bertscore import score
    # model_type=None uses BERTScore default (e.g. roberta-large). Optional: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" for medical.
    P, R, F1 = score(
        candidates,
        references,
        model_type=model_type,
        lang=lang,
        device=device,
        verbose=True,
    )
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def evaluate(model_path=None, dataset_name="medalpaca/medical_meadow_mediqa", val_split=0.1,
             max_eval_samples=None, max_new_tokens=256, temperature=0.1, bertscore_model=None):
    """
    Load model, run on validation set, compute BERTScore.
    """
    model_path = model_path or DEFAULT_MODEL_PATH
    from inference import load_model, generate_answer

    print("Loading model...")
    model, tokenizer = load_model(model_path)

    print("Loading evaluation dataset...")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    eval_dataset = load_eval_dataset(dataset_name, val_split=val_split, token=hf_token)
    n_total = len(eval_dataset)
    if max_eval_samples is not None and max_eval_samples < n_total:
        eval_dataset = eval_dataset.select(range(max_eval_samples))
        print(f"Using subset: {len(eval_dataset)} / {n_total} samples")
    else:
        print(f"Evaluating on {n_total} samples")

    candidates = []
    references = []
    for i, row in enumerate(eval_dataset):
        instruction = row.get("instruction", "")
        input_text = row.get("input", "") or ""
        ref = row.get("output", "")
        question = f"{instruction}\nContext: {input_text}" if input_text else instruction
        pred = generate_answer(
            model, tokenizer, instruction, context=input_text or None,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=0.95,
        )
        candidates.append(pred)
        references.append(ref)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{len(eval_dataset)}")

    print("\nComputing BERTScore...")
    device = next(model.parameters()).device
    metrics = run_bertscore(candidates, references, model_type=bertscore_model, device=str(device))

    print("\n" + "=" * 50)
    print("BERTScore metrics")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 50)
    return metrics


def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate medical Q&A model with BERTScore")
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Path to fine-tuned model")
    p.add_argument("--dataset", default="medalpaca/medical_meadow_mediqa", help="Dataset name")
    p.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio (must match training)")
    p.add_argument("--max_eval_samples", type=int, default=None, help="Cap number of eval samples (default: all)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--bertscore_model", default=None, help="BERTScore model (e.g. microsoft/deberta-xlarge-mnli)")
    args = p.parse_args()

    evaluate(
        model_path=args.model_path,
        dataset_name=args.dataset,
        val_split=args.val_split,
        max_eval_samples=args.max_eval_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        bertscore_model=args.bertscore_model,
    )


if __name__ == "__main__":
    main()
