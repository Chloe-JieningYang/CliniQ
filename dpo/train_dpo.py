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
from trl import DPOConfig, DPOTrainer


# ---------------------------------------------------------------------------
# Prompt helpers
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
    """Convert medAlpaca-style records into TRL DPOTrainer format.

    TRL DPOTrainer expects three keys:
        prompt   – the shared context / question
        chosen   – the preferred completion
        rejected – the dispreferred completion
    """
    prompt = build_prompt(
        instruction=example.get("instruction", ""),
        input_text=example.get("input", ""),
    )
    return {
        "prompt": prompt,
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    model: str,                 
    data_path: str = "dpo_medical_pairs.json",
    val_set_size: Union[int, float] = 0.05,
    output_dir: str = "./output/dpo",
    train_in_8bit: bool = False,
    train_in_4bit: bool = False,              
    fp16: bool = False,
    bf16: bool = True,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    sft_adapter_path: Optional[str] = None,
    beta: float = 0.1,                  
    loss_type: str = "sigmoid",       
    max_length: int = 1024,           
    num_epochs: int = 1,
    learning_rate: float = 5e-7,
    per_device_batch_size: int = 2,
    global_batch_size: int = 32,
    warmup_steps: int = 50,
    lr_scheduler_type: str = "cosine",
    optim: str = "adamw_torch",
    gradient_checkpointing: bool = True,  
    eval_steps: int = 100,
    save_total_limit: int = 3,
    use_wandb: bool = False,
    wandb_project: str = "medalpaca-dpo",
    wandb_run_name: str = "dpo-run",
    device_map: str = "auto",
    seed: int = 42,
):
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if fp16 and bf16:
        raise ValueError("Only one of fp16 / bf16 may be True.")
    if train_in_8bit and train_in_4bit:
        raise ValueError("Only one of train_in_8bit / train_in_4bit may be True.")
    if (train_in_8bit or train_in_4bit) and not use_lora:
        raise ValueError("Quantised training requires use_lora=True.")

    if use_wandb and wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size > 1
    gradient_accumulation_steps = max(1, global_batch_size // per_device_batch_size)
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

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

    print(f"Loading base model: {model}")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model_obj.config.use_cache = False

    if use_lora:
        if sft_adapter_path and os.path.exists(sft_adapter_path):
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

    ref_model = None


    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   

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

    dpo_args = DPOConfig(
        beta=beta,
        loss_type=loss_type,
        max_length=max_length,
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing, 
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
        report_to="wandb" if use_wandb else "none",
        run_name=wandb_run_name if use_wandb else None,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if ddp else None,
        seed=seed,
    )

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

    print("Starting DPO training…")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    fire.Fire(main)