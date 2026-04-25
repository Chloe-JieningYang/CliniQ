import json
import fire
from datasets import load_dataset

def convert_huggingface_to_eval(output_path="mediqa_eval_ready.json", max_samples=100):
    print("Fetching medalpaca/medical_meadow_mediqa from Hugging Face Hub...")
    

    dataset = load_dataset("medalpaca/medical_meadow_mediqa", split="train")
    
    eval_data = []

    for i, entry in enumerate(dataset):
        if max_samples > 0 and i >= max_samples:
            break
            
        eval_data.append({
            "instruction": entry["instruction"],
            "input": entry["input"],
            "reference": entry["output"]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Success! {len(eval_data)} samples saved to {output_path}")

if __name__ == "__main__":
    fire.Fire(convert_huggingface_to_eval)