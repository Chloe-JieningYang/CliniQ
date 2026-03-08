#!/usr/bin/env python3
"""
Medical Q&A inference using fine-tuned LoRA model
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def load_model(model_path="./medical_lora_output/final_model"):
    """Load fine-tuned model"""
    print(f"Loading model: {model_path}")
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_answer(model, tokenizer, question, context=None, max_length=512, temperature=0.7, top_p=0.9):
    """Generate medical Q&A answer"""
    if context:
        prompt = f"""Below is a medical question and answer.

Question: {question}
Context: {context}

Answer:"""
    else:
        prompt = f"""Below is a medical question and answer.

Question: {question}

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract answer part
    if "Answer:" in full_response:
        answer = full_response.split("Answer:")[-1].strip()
        # Remove end token
        answer = answer.replace("<|end_of_text|>", "").strip()
    else:
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer


def main():
    import sys
    
    # Model path
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./medical_lora_output/final_model"
    
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
