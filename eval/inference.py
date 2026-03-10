#!/usr/bin/env python3
"""
Medical Q&A inference using fine-tuned LoRA model
"""

import os
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Project root (parent of eval/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(ROOT, "medical_lora_output", "final_model")


def load_model(model_path=None):
    """Load fine-tuned model"""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    print(f"Loading model: {model_path}")
    
    # Use device_map="cuda:0" to avoid accelerate get_balanced_memory bug (unhashable set) with device_map="auto"
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Ensure tokenizer has proper settings
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_answer(model, tokenizer, question, context=None, max_new_tokens=256, temperature=0.3, top_p=0.95):
    """Generate medical Q&A answer using Llama 3 chat format"""
    # Build question text (with context if available)
    if context:
        question_text = f"{question}\nContext: {context}"
    else:
        question_text = question
    
    # Use Llama 3 chat format (same as training)
    prompt = f"""<|start_header_id|>user<|end_header_id|>
{question_text}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    # Get special tokens for stopping
    # Try to get eot_id token
    try:
        eot_token_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        if eot_token_id:
            eot_token_id = eot_token_id[0]  # Get the first token ID
        else:
            eot_token_id = None
    except Exception:
        eot_token_id = None
    
    eos_token_id = tokenizer.eos_token_id
    
    # Create stop token IDs list (use list for multiple stop tokens)
    stop_token_ids = []
    if eot_token_id is not None and eot_token_id != eos_token_id:
        stop_token_ids.append(eot_token_id)
    if eos_token_id is not None:
        stop_token_ids.append(eos_token_id)
    
    # Use the first stop token as eos_token_id, or use list if supported
    final_eos_token_id = stop_token_ids[0] if stop_token_ids else tokenizer.eos_token_id
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=final_eos_token_id,
            repetition_penalty=1.2,  # Increased repetition penalty to prevent loops
            no_repeat_ngram_size=3,  # Prevent repeating 3-grams
        )
    
    # Decode only the newly generated tokens (remove input prompt)
    generated_tokens = outputs[0][input_length:]
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    
    # Clean up the answer
    # Remove end tokens
    answer = answer.replace("<|eot_id|>", "").strip()
    answer = answer.replace("<|end_of_text|>", "").strip()
    
    # Remove any trailing special tokens or headers
    if "<|start_header_id|>" in answer:
        answer = answer.split("<|start_header_id|>")[0].strip()
    
    # If answer is empty or just whitespace, try extracting from full response
    if not answer or len(answer.strip()) < 5:
        answer_start_marker = "<|start_header_id|>assistant<|end_header_id|>\n"
        if answer_start_marker in full_response:
            answer = full_response.split(answer_start_marker)[-1].strip()
            answer = answer.replace("<|eot_id|>", "").strip()
            if "<|start_header_id|>" in answer:
                answer = answer.split("<|start_header_id|>")[0].strip()
    
    return answer


def main():
    import sys
    
    # Model path
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    print("\n" + "=" * 60)
    print("Medical Q&A Inference System")
    print("Type 'quit' or 'exit' to exit")
    print("=" * 60)
    
    # Interactive Q&A
    while True:
        question = input("\nEnter medical question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\nGenerating answer...")
        answer = generate_answer(model, tokenizer, question)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
