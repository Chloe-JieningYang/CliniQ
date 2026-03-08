#!/usr/bin/env python3
"""
Diagnose model training quality and check if LoRA weights are properly loaded
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import os

def diagnose_model(model_path="./medical_lora_output/final_model"):
    """Diagnose model issues"""
    
    print("=" * 60)
    print("Model Diagnosis")
    print("=" * 60)
    
    # Check if model files exist
    print("\n1. Checking model files...")
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  ✓ {file} exists ({size:.2f} MB)")
        else:
            print(f"  ✗ {file} MISSING!")
    
    # Try loading model
    print("\n2. Loading model...")
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("  ✓ Model loaded successfully")
        
        # Check LoRA adapters
        print("\n3. Checking LoRA adapters...")
        if hasattr(model, 'peft_config'):
            print(f"  ✓ LoRA adapters found: {len(model.peft_config)}")
            for name, config in model.peft_config.items():
                print(f"    - {name}: r={config.r}, alpha={config.lora_alpha}")
        else:
            print("  ✗ No LoRA adapters found!")
        
        # Check model parameters and LoRA weights
        print("\n4. Model parameters:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters (inference mode): {trainable_params:,}")
        print(f"  Note: In inference mode, requires_grad=False is normal")
        
        # Check LoRA adapter weights
        print("\n4.1. LoRA adapter weights:")
        if hasattr(model, 'peft_config'):
            for adapter_name, adapter_config in model.peft_config.items():
                # Count LoRA parameters
                lora_params = 0
                for name, module in model.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        lora_A_params = sum(p.numel() for p in module.lora_A.parameters())
                        lora_B_params = sum(p.numel() for p in module.lora_B.parameters())
                        lora_params += lora_A_params + lora_B_params
                
                print(f"  LoRA adapter '{adapter_name}':")
                print(f"    - r={adapter_config.r}, alpha={adapter_config.lora_alpha}")
                print(f"    - LoRA parameters: {lora_params:,}")
                print(f"    - Expected LoRA params (approx): ~{adapter_config.r * 2 * 8 * 1024 * 1024:,} (r*2*8M for 8B model)")
                
                if lora_params == 0:
                    print("    ⚠ WARNING: No LoRA weights found! Model may not be using LoRA adapters.")
                else:
                    print(f"    ✓ LoRA weights are loaded and active")
        
        # Test a simple generation
        print("\n5. Testing generation...")
        test_prompt = "<|start_header_id|>user<|end_header_id|>\nWhat is diabetes?\n<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # Very low temperature for testing
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"  Generated text (first 200 chars):")
        print(f"  {generated[:200]}...")
        
        # Check if it's just repeating or generating nonsense
        if len(set(generated.split()[:10])) < 3:
            print("  ⚠ Warning: Model seems to be repeating tokens")
        if any(char in generated for char in ['<|', 'eot_id', 'start_header']):
            print("  ⚠ Warning: Model may not be stopping properly")
        
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Diagnosis complete")
    print("=" * 60)
    
    print("\nRecommendations:")
    print("1. Check TensorBoard logs to see if training loss decreased")
    print("2. Verify training data quality")
    print("3. Consider re-training with:")
    print("   - More epochs (5-10 instead of 3)")
    print("   - Lower learning rate (1e-4 instead of 2e-4)")
    print("   - More training data")
    print("4. Check if base model (Llama-3.1-8B-Instruct) works correctly")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./medical_lora_output/final_model"
    diagnose_model(model_path)
