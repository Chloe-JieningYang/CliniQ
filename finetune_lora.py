#!/usr/bin/env python3
"""
LoRA fine-tuning for Llama models for medical Q&A
Using medalpaca/medical_meadow_mediqa dataset
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_and_prepare_dataset(dataset_name="medalpaca/medical_meadow_mediqa", token=None, val_split=0.1):
    """Load and prepare medical Q&A dataset"""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train", token=token)
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}")
    
    # View a sample
    if len(dataset) > 0:
        print("\nSample data:")
        print(dataset[0])
    
    # Split into train and validation sets
    if val_split > 0:
        dataset = dataset.train_test_split(test_size=val_split, seed=42)
        print(f"\nTrain set size: {len(dataset['train'])}")
        print(f"Validation set size: {len(dataset['test'])}")
        return dataset['train'], dataset['test']
    else:
        return dataset, None


def formatting_func(examples):
    """Convert medical Q&A data to Llama 3 chat format
    Format: <|start_header_id|>user<|end_header_id|>\nquestion\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\nanswer\n<|eot_id|>
    Only answer part will be used for loss calculation (via labels masking)
    """
    output_texts = []
    for i in range(len(examples.get("instruction", []))):
        instruction = examples["instruction"][i] if "instruction" in examples else ""
        output = examples["output"][i] if "output" in examples else ""
        input_text = examples.get("input", [""] * len(examples["instruction"]))[i] if "input" in examples else ""
        
        # Build question text (with context if available)
        if input_text:
            question = f"{instruction}\nContext: {input_text}"
        else:
            question = instruction
        
        # Use Llama 3 chat format
        text = f"""<|start_header_id|>user<|end_header_id|>
{question}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
{output}
<|eot_id|>"""
        
        output_texts.append(text)
    
    return {"text": output_texts}


def main():
    # ============ Configuration Parameters ============
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_name = "medalpaca/medical_meadow_mediqa"
    output_dir = "./medical_lora_output"
    max_seq_length = 512
    val_split = 0.1  # 10% for validation
    
    # ============ Get Hugging Face Token ============
    # Read from .env file or environment variables
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if not hf_token:
        print("=" * 60)
        print("Warning: Hugging Face Token not found!")
        print("Llama models require access permission. Please set token in one of:")
        print("  1. .env file: HF_TOKEN=your_token_here")
        print("  2. Environment variable: export HF_TOKEN=your_token_here")
        print("=" * 60)
        print("\nHow to get Token:")
        print("1. Visit https://huggingface.co/settings/tokens")
        print("2. Create a new token (requires 'Read' permission)")
        print(f"3. Visit https://huggingface.co/{model_id} and accept terms")
        print("4. Add to .env file: HF_TOKEN=your_token_here")
        print("\nContinue? (If model is already cached locally, token may not be needed)")
        response = input("Continue? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting. Please set HF_TOKEN in .env file or environment variable first.")
            return
    else:
        # Login with token
        print("Authenticating with Hugging Face Token...")
        try:
            login(token=hf_token, add_to_git_credential=False)
            print("Authentication successful!")
        except Exception as e:
            print(f"Authentication failed: {e}")
            print("Please check if token is correct")
            return
    
    # LoRA configuration
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    
    # Training parameters
    # Note: Adjust batch size based on model size and available GPU memory
    num_train_epochs = 3
    per_device_train_batch_size = 2  # Adjust based on model size and GPU memory
    gradient_accumulation_steps = 8  # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
    learning_rate = 2e-4  # learning rate
    warmup_steps = 100  # warmup steps
    
    print("=" * 60)
    print("Medical Q&A LoRA Fine-tuning Pipeline")
    print("=" * 60)
    
    # ============ 1. Load Dataset ============
    train_dataset, eval_dataset = load_and_prepare_dataset(dataset_name, token=hf_token, val_split=val_split)
    
    # ============ 2. Configure Quantization (Save Memory) ============
    print("\nConfiguring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # ============ 3. Load Model and Tokenizer ============
    print(f"\nLoading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    
    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Loading model (4-bit quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        token=hf_token,  # Use token to access restricted model
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # ============ 4. Configure LoRA ============
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Llama attention modules
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    print("\nTrainable parameters statistics:")
    model.print_trainable_parameters()
    
    # ============ 5. Process Dataset ============
    print("\nProcessing dataset...")
    # Format training dataset with Llama 3 chat format
    train_dataset = train_dataset.map(formatting_func, batched=True, remove_columns=train_dataset.column_names)
    
    # Format validation dataset if exists
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(formatting_func, batched=True, remove_columns=eval_dataset.column_names)
    
    # Tokenize and create labels (mask everything except answer part)
    def tokenize_and_mask(examples):
        # Tokenize the text (no padding here, let data collator handle it)
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,  # No padding - data collator will handle it
        )
        
        # Create labels: mask everything except the answer part
        # Answer part starts after "<|start_header_id|>assistant<|end_header_id|>\n"
        labels = []
        for i, text in enumerate(examples["text"]):
            # Get the tokenized input_ids for this example
            input_ids = tokenized["input_ids"][i]
            input_length = len(input_ids)
            
            # Find where answer starts in the original text
            answer_start_marker = "<|start_header_id|>assistant<|end_header_id|>\n"
            answer_start_idx = text.find(answer_start_marker)
            
            if answer_start_idx != -1:
                # Tokenize the part before answer (should be masked)
                before_answer = text[:answer_start_idx + len(answer_start_marker)]
                before_tokens = tokenizer.encode(before_answer, add_special_tokens=False)
                before_length = len(before_tokens)
                
                # Create labels: -100 for masked parts, token_id for answer part
                # Match the length of input_ids (which may be truncated)
                label = [-100] * input_length
                
                # Only the answer part (after assistant header) should have labels
                # Make sure we don't exceed input_length
                answer_start_pos = min(before_length, input_length)
                for j in range(answer_start_pos, input_length):
                    # Use the actual token ID from input_ids
                    label[j] = input_ids[j]
            else:
                # Fallback: if format is wrong, mask everything
                label = [-100] * input_length
            
            labels.append(label)
        
        # Set labels (no padding - data collator will handle it during batching)
        tokenized["labels"] = labels
        return tokenized
    
    print("Tokenizing dataset and creating labels (masking question part, only answer will be trained)...")
    train_dataset = train_dataset.map(
        tokenize_and_mask,
        batched=True,
        remove_columns=["text"]
    )
    
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            tokenize_and_mask,
            batched=True,
            remove_columns=["text"]
        )
    
    # Show processed sample
    if len(train_dataset) > 0:
        print("\nProcessed data sample:")
        print(f"Input IDs length: {len(train_dataset[0]['input_ids'])}")
        print(f"Labels length: {len(train_dataset[0]['labels'])}")
        # Count how many tokens are not masked
        non_masked = sum(1 for l in train_dataset[0]['labels'] if l != -100)
        print(f"Non-masked tokens (answer part): {non_masked} out of {len(train_dataset[0]['labels'])}")
    
    # ============ 6. Training Arguments ============
    print("\nConfiguring training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,  # L4 GPU supports bfloat16
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        optim="paged_adamw_8bit",  # 8-bit optimizer, saves memory
        lr_scheduler_type="cosine",
        report_to="tensorboard",  # Use TensorBoard for visualization
        logging_dir=os.path.join(output_dir, "logs"),  # TensorBoard log directory
        remove_unused_columns=False,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=50 if eval_dataset is not None else None,
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
    )
    
    # ============ 7. Initialize SFTTrainer ============
    print("\nInitializing trainer...")
    # Important: We've already tokenized data and set labels manually
    # Labels with -100 will be ignored in loss calculation (only answer part will be trained)
    # SFTTrainer's data collator will handle padding during batching
    # The labels will be padded to match input_ids by the collator
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,  # Already tokenized with labels
        eval_dataset=eval_dataset,  # Already tokenized with labels
        args=training_args,
        processing_class=tokenizer,  # Needed for padding/collation during batching
        # No formatting_func needed since data is already tokenized
    )
    
    # ============ 8. Start Fine-tuning ============
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()
    
    # ============ 9. Save Fine-tuned Model ============
    print("\nSaving model...")
    final_model_dir = os.path.join(output_dir, "final_model")
    
    # Save LoRA adapter (this saves only the adapter weights, not the full model)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Verify that adapter was saved
    adapter_path = os.path.join(final_model_dir, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        adapter_size = os.path.getsize(adapter_path) / (1024 * 1024)  # MB
        print(f"  ✓ LoRA adapter saved: {adapter_size:.2f} MB")
    else:
        print("  ⚠ Warning: adapter_model.safetensors not found!")
    
    # ============ 10. Training Summary ============
    print(f"\nTraining completed! Model saved to: {final_model_dir}")
    
    # Find actual TensorBoard log directory (may be in runs or logs)
    log_dir = os.path.join(output_dir, "logs")
    runs_dir = os.path.join(output_dir, "runs")
    
    # Check which directory exists and has log files
    actual_log_dir = None
    if os.path.exists(runs_dir) and os.listdir(runs_dir):
        # Find the most recent run directory
        run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if run_dirs:
            actual_log_dir = os.path.join(runs_dir, run_dirs[-1])  # Use most recent
    elif os.path.exists(log_dir):
        actual_log_dir = log_dir
    
    if actual_log_dir:
        print(f"TensorBoard logs saved to: {actual_log_dir}")
        print("\nTo view TensorBoard, run:")
        print(f"  tensorboard --logdir {actual_log_dir}")
        print("Or to view all runs:")
        print(f"  tensorboard --logdir {output_dir}")
        print("Then open http://localhost:6006 in your browser")
    else:
        print(f"TensorBoard logs directory: {log_dir} (check if logs were generated)")
    print("=" * 60)


if __name__ == "__main__":
    main()
