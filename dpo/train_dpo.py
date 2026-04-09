"""
DPO (Direct Preference Optimization) training script for medAlpaca.

Extends the original medAlpaca SFT pipeline with DPO fine-tuning using
TRL's DPOTrainer. Assumes the model has already been SFT-trained (or you
can start from any causal-LM checkpoint).

Data format expected (JSON / JSONL), one record per line:
    {
        "instruction": "What are symptoms of diabetes?",
        "input": "",                          # optional context
        "chosen": "Common symptoms include ...",
        "rejected": "I don't know."
    }

Usage:
    # basic run
    python train_dpo.py \
        --model ./output/sft-checkpoint \
        --data_path dpo_medical_pairs.json \
        --output_dir ./output/dpo

    # with LoRA + 4-bit quantisation
    python train_dpo.py \
        --model meta-llama/Llama-2-7b-hf \
        --data_path dpo_medical_pairs.json \
        --output_dir ./output/dpo \
        --use_lora True \
        --train_in_4bit True \
        --bf16 True
"""

import os
import sys
from typing import List, Optional, Tuple, Union

import fire
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import torch.distributed._composable.fsdp
# Manually inject the missing class into the namespace trl expects
import torch.distributed.fsdp
if not hasattr(torch.distributed.fsdp, "FSDPModule"):
    from torch.distributed._composable.fsdp import FSDPModule
    torch.distributed.fsdp.FSDPModule = FSDPModule
from trl import DPOConfig, DPOTrainer


# ---------------------------------------------------------------------------
# Prompt helpers (mirrors medAlpaca's DataHandler prompt format)
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


def preprocess_dpo(example: dict) -> dict:
    return {
        "prompt": example["prompt"], 
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    # ── Model ────────────────────────────────────────────────────────────────
    model: str,                                   # HF hub id or local path
    # ── Data ─────────────────────────────────────────────────────────────────
    data_path: str = "dpo_medical_pairs.json",
    val_set_size: Union[int, float] = 0.05,
    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "./output/dpo",
    # ── Precision / quantisation ──────────────────────────────────────────────
    train_in_8bit: bool = False,
    train_in_4bit: bool = False,                  # requires bitsandbytes >= 0.39
    fp16: bool = False,
    bf16: bool = True,
    # ── LoRA ─────────────────────────────────────────────────────────────────
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    # ── SFT adapter (optional) ────────────────────────────────────────────────
    sft_adapter_path: Optional[str] = None,
    # ── DPO hyper-parameters ──────────────────────────────────────────────────
    beta: float = 0.1,                            # KL penalty coefficient
    loss_type: str = "sigmoid",                   # "sigmoid" | "hinge" | "ipo" | "kto_pair"
    max_length: int = 1024,                       # prompt + completion
    # ── Training ─────────────────────────────────────────────────────────────
    num_epochs: int = 1,
    learning_rate: float = 5e-7,
    per_device_batch_size: int = 2,
    global_batch_size: int = 32,
    warmup_steps: int = 50,
    lr_scheduler_type: str = "cosine",
    optim: str = "adamw_torch",
    gradient_checkpointing: bool = True,  
    # ── Evaluation & saving ───────────────────────────────────────────────────
    eval_steps: int = 100,
    save_total_limit: int = 3,
    # ── Logging ───────────────────────────────────────────────────────────────
    use_wandb: bool = False,
    wandb_project: str = "medalpaca-dpo",
    wandb_run_name: str = "dpo-run",
    # ── Misc ─────────────────────────────────────────────────────────────────
    device_map: str = "auto",
    seed: int = 42,
):
    # ── Defaults ──────────────────────────────────────────────────────────────
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # ── Validation ────────────────────────────────────────────────────────────
    if fp16 and bf16:
        raise ValueError("Only one of fp16 / bf16 may be True.")
    if train_in_8bit and train_in_4bit:
        raise ValueError("Only one of train_in_8bit / train_in_4bit may be True.")
    if (train_in_8bit or train_in_4bit) and not use_lora:
        raise ValueError("Quantised training requires use_lora=True.")

    # ── W&B ──────────────────────────────────────────────────────────────────
    if use_wandb and wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    # ── Distributed setup ─────────────────────────────────────────────────────
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size > 1
    gradient_accumulation_steps = max(1, global_batch_size // per_device_batch_size)
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

    # ── Quantisation config ────────────────────────────────────────────────────
    bnb_config = None
    if train_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif train_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"Loading base model: {model}")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model_obj.config.use_cache = False

    # ── Attach LoRA / SFT adapters ────────────────────────────────────────────
    if use_lora:
        if sft_adapter_path and os.path.exists(sft_adapter_path):
            # Load existing SFT adapters and keep them trainable for DPO.
            print(f"Loading SFT adapters from {sft_adapter_path}…")
            model_obj = PeftModel.from_pretrained(
                model_obj, sft_adapter_path, is_trainable=True
            )
        else:
            if sft_adapter_path:
                print(f"SFT adapter path '{sft_adapter_path}' not found — initialising fresh LoRA.")
            else:
                print("No SFT adapter path provided — initialising fresh LoRA.")

            if train_in_8bit or train_in_4bit:
                model_obj = prepare_model_for_kbit_training(model_obj)

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model_obj = get_peft_model(model_obj, lora_config)
            model_obj.print_trainable_parameters()

    # ── Reference model ───────────────────────────────────────────────────────
    # TRL automatically disables LoRA adapters on a PeftModel to derive the
    # reference policy, so ref_model=None is correct in both branches above.
    ref_model = None

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   # causal LM convention

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading dataset from: {data_path}")
    raw = load_dataset("json", data_files=data_path, split="train")
    raw = raw.map(preprocess_dpo, remove_columns=raw.column_names)

    if val_set_size > 0:
        split = raw.train_test_split(test_size=val_set_size, seed=seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = raw
        eval_dataset = None

    print(f"  Train samples : {len(train_dataset)}")
    if eval_dataset:
        print(f"  Eval  samples : {len(eval_dataset)}")

    # ── DPO training arguments ────────────────────────────────────────────────
    dpo_args = DPOConfig(
        beta=beta,
        loss_type=loss_type,
        max_length=max_length,
        # standard training args
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,  # FIX: added
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        optim=optim,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=(eval_dataset is not None),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        report_to=["tensorboard", "wandb"] if use_wandb else ["tensorboard"],
        logging_dir=f"{output_dir}/logs",
        run_name=wandb_run_name if use_wandb else None,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if ddp else None,
        seed=seed,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    use_compile = (
        torch.__version__ >= "2"
        and sys.platform != "win32"
        and not (train_in_8bit or train_in_4bit)
    )
    if use_compile:
        print("Compiling model with torch.compile…")
        model_obj = torch.compile(model_obj)

    trainer = DPOTrainer(
        model=model_obj,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting DPO training…")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    fire.Fire(main)