import json
import fire
from datasets import load_dataset

def convert_huggingface_to_eval(max_samples=100):
    print("Fetching medalpaca/medical_meadow_mediqa from Hugging Face Hub...")
    
    # Load the dataset
    dataset = load_dataset(
        "json", 
        data_files="./eval_set/medical_meadow_small.json", 
        split="train"
    )
    
    patient_data = []
    doctor_data = []
    
    for i, entry in enumerate(dataset):
        if max_samples > 0 and i >= max_samples:
            break
            
        # We prepend the persona to the instruction, 
        # but keep the actual question in the 'input' field.
        
        # Create Patient entry
        patient_data.append({
            "instruction": f"I am a patient.",
            "input": entry["input"],
            "reference": entry["output"]
        })

        # Create Doctor entry
        doctor_data.append({
            "instruction": f"I am a doctor.",
            "input": entry["input"],
            "reference": entry["output"]
        })

    # Save Patient file
    patient_path = "./eval_set/patient_eval.json"
    with open(patient_path, "w", encoding="utf-8") as f:
        json.dump(patient_data, f, indent=4, ensure_ascii=False)

    # Save Doctor file
    doctor_path = "./eval_set/doctor_eval.json"
    with open(doctor_path, "w", encoding="utf-8") as f:
        json.dump(doctor_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Success!")
    print(f"Saved {len(patient_data)} samples to {patient_path}")
    print(f"Saved {len(doctor_data)} samples to {doctor_path}")

if __name__ == "__main__":
    fire.Fire(convert_huggingface_to_eval)