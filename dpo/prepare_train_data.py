"""
prepare_dpo_data.py
====================
Utility to create DPO preference pairs from medAlpaca-style datasets.
No external API required — everything runs locally.

Three modes:
1. MANUAL  (--mode manual)
   You already have chosen/rejected columns — just validates & normalises.
2. RULE  (--mode rule)   <- simplest, no GPU needed
   Degrades the original answer with deterministic text transformations
   (truncation, sentence shuffling, filler injection, etc.) to produce
   "rejected" responses.
3. MODEL  (--mode model)   <- highest quality
   Loads your SFT checkpoint locally and samples a second response at high
   temperature to use as the "rejected" answer.
   Requires: pip install transformers accelerate

Output schema (one JSON object per line):
    {
        "instruction": "...",
        "input":       "...",   # may be empty
        "chosen":      "...",   # preferred answer  (original SFT output)
        "rejected":    "..."    # dispreferred answer
    }

Usage examples:
    # Validate an existing pairs file
    python prepare_dpo_data.py \\
        --mode manual \\
        --input_path my_pairs.jsonl \\
        --output_path dpo_pairs_manual.json

    # Rule-based degradation (no GPU)
    python prepare_dpo_data.py \\
        --mode rule \\
        --input_path medical_meadow_small.json \\
        --output_path dpo_pairs_rule.json

    # Model-based (uses your SFT checkpoint)
    python prepare_dpo_data.py \\
        --mode model \\
        --input_path medical_meadow_small.json \\
        --output_path dpo_pairs_llama.json \\
        --model_path ./output/sft-checkpoint \\
        --max_samples 2000
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import List

import fire
from datasets import load_dataset


# ---------------------------------------------------------------------------
# FIX: moved dataset download out of module-level scope and into a
# standalone helper so it only runs when explicitly called, not on every
# import or subcommand invocation.
# ---------------------------------------------------------------------------
def _download_sample_dataset(output_path: str = "medical_meadow_small.json"):
    """Download the wikidoc patient-information split and save as JSONL."""
    dataset = load_dataset(
        "medalpaca/medical_meadow_wikidoc_patient_information", split="train"
    )
    dataset.to_json(output_path)
    print(f"File saved as {output_path}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_jsonl_or_json(path: str) -> list:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def save_jsonl(records: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records → {path}")


# ---------------------------------------------------------------------------
# Mode: manual  (validate + normalise)
# ---------------------------------------------------------------------------

def mode_manual(input_path: str, output_path: str):
    records = load_jsonl_or_json(input_path)
    out, skipped = [], 0
    for r in records:
        if "chosen" not in r or "rejected" not in r:
            skipped += 1
            continue
        out.append({
            "instruction": r.get("instruction", ""),
            "input":       r.get("input", ""),
            "chosen":      r["chosen"].strip(),
            "rejected":    r["rejected"].strip(),
        })
    print(f"Valid pairs: {len(out)}  |  Skipped (missing keys): {skipped}")
    save_jsonl(out, output_path)


# ---------------------------------------------------------------------------
# Mode: rule  (deterministic text degradations — no GPU)
# ---------------------------------------------------------------------------

FILLER_PHRASES = [
    "It depends on the situation.",
    "This is a complex topic.",
    "You should consult a doctor.",
    "There are many factors involved.",
    "Medical advice varies.",
]

HEDGE_PREFIXES = [
    "I'm not entirely sure, but ",
    "This might not be accurate, but ",
    "Generally speaking, ",
    "It's hard to say definitively, but ",
]


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter that preserves trailing whitespace."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


def degrade_truncate(text: str, rng: random.Random) -> str:
    """Keep only the first 30-60 % of the answer."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        words = text.split()
        keep = max(1, int(len(words) * rng.uniform(0.3, 0.5)))
        return " ".join(words[:keep]) + "..."
    keep = max(1, int(len(sentences) * rng.uniform(0.3, 0.6)))
    return " ".join(sentences[:keep]) + "..."


def degrade_shuffle(text: str, rng: random.Random) -> str:
    """Shuffle the middle sentences, keeping first and last."""
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return degrade_truncate(text, rng)
    middle = sentences[1:-1]
    rng.shuffle(middle)
    return " ".join([sentences[0]] + middle + [sentences[-1]])


def degrade_hedge(text: str, rng: random.Random) -> str:
    """Prepend a hedging phrase and append a vague filler sentence."""
    prefix = rng.choice(HEDGE_PREFIXES)
    suffix = rng.choice(FILLER_PHRASES)
    return f"{prefix}{text} {suffix}"


def degrade_drop_details(text: str, rng: random.Random) -> str:
    """Remove sentences that contain numbers, dosages, or specific terms."""
    sentences = _split_sentences(text)
    filtered = [
        s for s in sentences
        if not re.search(r'\d+\s*(?:mg|ml|%|mmHg|g|mcg|units?)', s, re.I)
    ]
    # FIX: fall back to truncation only when ALL sentences were removed,
    # not just when the filtered list happens to be falsy for other reasons.
    if len(filtered) == 0:
        return degrade_truncate(text, rng)
    return " ".join(filtered)


DEGRADATION_FNS = [degrade_truncate, degrade_shuffle, degrade_hedge, degrade_drop_details]


def rule_degrade(text: str, rng: random.Random) -> str:
    """Apply 1-2 random degradations."""
    fns = rng.sample(DEGRADATION_FNS, k=rng.randint(1, 2))
    result = text
    for fn in fns:
        result = fn(result, rng)
    return result.strip()


def mode_rule(
    input_path: str,
    output_path: str,
    max_samples: int = 0,
    seed: int = 42,
):
    records = load_jsonl_or_json(input_path)
    rng = random.Random(seed)
    rng.shuffle(records)
    if max_samples > 0:
        records = records[:max_samples]

    out = []
    for i, r in enumerate(records, 1):
        instruction = r.get("instruction", "")
        input_text  = r.get("input", "")
        chosen      = r.get("output", r.get("chosen", "")).strip()
        if not chosen:
            continue

        rejected = rule_degrade(chosen, rng)
        # Sanity-check: rejected must differ from chosen
        if rejected == chosen:
            rejected = degrade_truncate(chosen, rng)

        out.append({
            "instruction": instruction,
            "input":       input_text,
            "chosen":      chosen,
            "rejected":    rejected,
        })
        if i % 100 == 0:
            print(f"  Processed {i}/{len(records)}…")

    print(f"Generated {len(out)} pairs via rule-based degradation.")
    save_jsonl(out, output_path)


# ---------------------------------------------------------------------------
# Mode: model  (sample from SFT checkpoint at high temperature)
# ---------------------------------------------------------------------------

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def build_prompt(instruction: str, input_text: str = "") -> str:
    if input_text and input_text.strip():
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_WITHOUT_INPUT.format(instruction=instruction)

import torch

def _generate_batch(model, tokenizer, prompts, max_new_tokens, temperature, top_p=0.9):
    """Run one batched generation pass; returns a list of decoded strings."""
    enc = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=512,
    ).to(model.device)
    input_len = enc["input_ids"].shape[1]
    greedy = temperature == 0.0
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            top_p=None if greedy else top_p,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)


def _clean(text: str) -> str:
    """Strip model artifacts from a generated response."""
    for phrase in ("Answer this question truthfully",
                   "Your task requires truthful responses"):
        text = text.replace(phrase, "")
    return text.strip()


def _load_sft_model(model_path: str, base_model_id: str, load_in_4bit: bool = False):
    """Load base model + SFT adapter, merge, and dispatch. Shared by model modes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from accelerate import dispatch_model, infer_auto_device_map

    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading base model: {base_model_id} (4-bit={load_in_4bit})")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if not load_in_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading and merging SFT adapter from: {model_path}")
    if load_in_4bit:
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=False)
    else:
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=False)
        model = model.merge_and_unload()
        no_split = [
            "LlamaDecoderLayer", "MistralDecoderLayer",
            "Qwen2DecoderLayer", "FalconDecoderLayer",
            "GPTNeoXLayer", "BloomBlock",
        ]
        device_map = infer_auto_device_map(model, no_split_module_classes=no_split)
        model = dispatch_model(model, device_map=device_map)

    model.eval()
    return model, tokenizer


def mode_model(
    input_path: str,
    output_path: str,
    model_path: str = "",
    base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_samples: int = 1000,
    # ── Temperature for each side ────────────────────────────────────────────
    chosen_temperature: float = 0.0,    # 0.0 = greedy/deterministic → clean chosen
    rejected_temperature: float = 1.4,  # high temp → diverse, lower-quality rejected
    # ────────────────────────────────────────────────────────────────────────
    max_new_tokens: int = 256,
    batch_size: int = 4,
    seed: int = 42,
    min_rejected_length: int = 10,      # skip pairs where rejected is suspiciously short
    load_in_4bit: bool = False,
):
    """
    Both chosen and rejected are generated by the model, but at different
    temperatures:
      chosen   — greedy / low temperature  → the model's best answer
      rejected — high temperature          → a diverse, lower-quality sample

    This creates a clear quality gap between the two sides without relying on
    any external gold annotations.
    """
    if not model_path:
        sys.exit("--model_path is required for mode=model.")
    if chosen_temperature == rejected_temperature:
        print("⚠  Warning: chosen_temperature == rejected_temperature. "
              "The DPO signal will be weak — consider widening the gap.")

    from transformers import set_seed
    set_seed(seed)

    model, tokenizer = _load_sft_model(model_path, base_model_id, load_in_4bit)

    # Load data
    records = load_jsonl_or_json(input_path)
    random.seed(seed)
    random.shuffle(records)
    if max_samples > 0:
        records = records[:max_samples]
    print(f"Loaded {len(records)} records.")
    print(f"chosen_temperature={chosen_temperature}  "
          f"rejected_temperature={rejected_temperature}")

    out = []
    skipped = 0

    for batch_start in range(0, len(records), batch_size):
        batch   = records[batch_start: batch_start + batch_size]
        prompts = []
        metas   = []

        for r in batch:
            instruction = r.get("instruction", r.get("question", ""))
            input_text  = r.get("input", "")
            prompts.append(build_prompt(instruction, input_text))
            metas.append({"instruction": instruction, "input": input_text})

        if not prompts:
            continue

        # ── Generate chosen at low/zero temperature (best answer) ─────────────
        chosen_texts = _generate_batch(
            model, tokenizer, prompts,
            max_new_tokens=max_new_tokens,
            temperature=chosen_temperature,
        )

        # ── Generate rejected at high temperature (diverse / weaker answer) ───
        rejected_texts = _generate_batch(
            model, tokenizer, prompts,
            max_new_tokens=max_new_tokens,
            temperature=rejected_temperature,
        )

        for meta, chosen_raw, rejected_raw in zip(metas, chosen_texts, rejected_texts):
            chosen   = _clean(chosen_raw)
            rejected = _clean(rejected_raw)

            # Skip pairs with no usable signal
            if len(rejected.split()) < min_rejected_length:
                skipped += 1
                continue
            if chosen.strip().lower() == rejected.strip().lower():
                skipped += 1
                continue

            # print("\n==============================")
            # print("Instruction:", meta["instruction"])
            # print(f"\nChosen (temp={chosen_temperature}):")
            # print(chosen)
            # print(f"\nRejected (temp={rejected_temperature}):")
            # print(rejected)
            # print("==============================\n")

            out.append({
                "instruction": meta["instruction"],
                "input":       meta["input"],
                "chosen":      chosen,
                "rejected":    rejected,
            })

        print(f"  [{batch_start + len(batch)}/{len(records)}] processed  "
              f"| pairs so far: {len(out)}  skipped: {skipped}…")

    print(f"\nDone. Generated {len(out)} pairs  |  Skipped {skipped} "
          f"(too short or identical).")
    save_jsonl(out, output_path)



# ---------------------------------------------------------------------------
# Mode: model-rejected  (chosen = original dataset output,
#                        rejected = model sample at high temperature)
#
# Key difference from mode_model:
#   mode_model       — both chosen AND rejected come from the model
#                      (chosen = greedy/low-temp, rejected = high-temp)
#   model-rejected   — chosen = gold answer from the dataset (no generation)
#                      rejected = model sample at high temperature
#
# This is the recommended strategy when your dataset already has high-quality
# gold answers and you want the DPO signal to push away from model hallucinations.
# ---------------------------------------------------------------------------

def mode_model_rejected(
    input_path: str,
    output_path: str,
    model_path: str = "",
    base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_samples: int = 1000,
    temperature: float = 1.4,       # high temp → diverse / lower-quality rejected
    max_new_tokens: int = 256,
    batch_size: int = 4,
    seed: int = 42,
    min_rejected_length: int = 10,  # skip pairs where rejected is suspiciously short
    load_in_4bit: bool = False,
):
    """
    Build DPO pairs where:
      chosen   = the original gold answer from the dataset (field: output/chosen/answer)
      rejected = a sample generated by the model at high temperature

    Unlike mode_model, this never generates the chosen response — it trusts the
    dataset annotation directly, which is faster and avoids contaminating the
    chosen side with model outputs.
    """
    if not model_path:
        sys.exit("--model_path is required for mode=model-rejected.")

    from transformers import set_seed
    set_seed(seed)

    model, tokenizer = _load_sft_model(model_path, base_model_id, load_in_4bit)

    # Load + shuffle data
    records = load_jsonl_or_json(input_path)
    random.seed(seed)
    random.shuffle(records)

    # Keep only records that have a non-empty gold answer
    records = [
        r for r in records
        if r.get("output", r.get("chosen", r.get("answer", ""))).strip()
    ]
    if max_samples > 0:
        records = records[:max_samples]
    print(f"Loaded {len(records)} records with valid gold answers.")

    out = []
    skipped = 0

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start: batch_start + batch_size]
        prompts, chosens, metas = [], [], []

        for r in batch:
            instruction = r.get("instruction", r.get("question", ""))
            input_text  = r.get("input", "")
            # chosen = gold answer straight from the dataset — no generation needed
            chosen = r.get("output", r.get("chosen", r.get("answer", ""))).strip()
            prompts.append(build_prompt(instruction, input_text))
            chosens.append(chosen)
            metas.append({"instruction": instruction, "input": input_text})

        if not prompts:
            continue

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.8,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)

        for meta, chosen, rejected_raw in zip(metas, chosens, decoded):
            # Clean up model artifacts
            rejected = rejected_raw.replace("Answer this question truthfully", "")
            rejected = rejected.replace("Your task requires truthful responses", "")
            rejected = rejected.strip()

            # Skip pairs where the model produced an empty or trivially short response
            if len(rejected.split()) < min_rejected_length:
                skipped += 1
                continue

            # Skip pairs where model accidentally reproduced the gold answer verbatim
            if rejected.strip().lower() == chosen.strip().lower():
                skipped += 1
                continue

            # print("\n==============================")
            # print("Instruction:", meta["instruction"])
            # print("\nChosen (gold dataset answer):")
            # print(chosen)
            # print("\nRejected (model sample):")
            # print(rejected)
            # print("==============================\n")

            out.append({
                "instruction": meta["instruction"],
                "input":       meta["input"],
                "chosen":      chosen,
                "rejected":    rejected,
            })

        print(f"  [{batch_start + len(batch)}/{len(records)}] processed…")

    print(f"\nGenerated {len(out)} pairs  |  Skipped {skipped} "
          f"(too short or identical to chosen).")
    save_jsonl(out, output_path)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    mode: str = "rule",                       # manual | rule | model | model-rejected
    input_path: str = "",
    output_path: str = "dpo_medical_pairs.json",
    # rule / model shared
    max_samples: int = 0,
    seed: int = 42,
    # model / model-rejected mode
    model_path: str = "",
    base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    chosen_temperature: float = 0.0,    # mode=model only: temp for chosen generation
    rejected_temperature: float = 1.4,  # mode=model / model-rejected: temp for rejected
    max_new_tokens: int = 256,
    batch_size: int = 4,
    load_in_4bit: bool = False,
):
    if not input_path:
        sys.exit("--input_path is required.")

    if mode == "manual":
        mode_manual(input_path, output_path)
    elif mode == "rule":
        mode_rule(input_path, output_path, max_samples=max_samples, seed=seed)
    elif mode == "model":
        mode_model(
            input_path, output_path,
            model_path=model_path,
            base_model_id=base_model_id,
            max_samples=max_samples or 1000,
            chosen_temperature=chosen_temperature,
            rejected_temperature=rejected_temperature,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            seed=seed,
            load_in_4bit=load_in_4bit,
        )
    elif mode == "model-rejected":
        mode_model_rejected(
            input_path, output_path,
            model_path=model_path,
            base_model_id=base_model_id,
            max_samples=max_samples or 1000,
            temperature=rejected_temperature,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            seed=seed,
            load_in_4bit=load_in_4bit,
        )
    else:
        sys.exit(f"Unknown mode: {mode!r}. Choose 'manual', 'rule', 'model', or 'model-rejected'.")


if __name__ == "__main__":
    # Pass main as the sole target so Fire never builds a multi-command tree,
    # even when other callables (like download_sample_dataset) exist at module level.
    fire.Fire(main)