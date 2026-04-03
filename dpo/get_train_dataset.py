from datasets import load_dataset

def fetch_medical_meadow():
    print("Fetching Technical Wikidoc dataset...")
    # 1. Technical/Professional dataset
    tech_ds = load_dataset("medalpaca/medical_meadow_wikidoc", split="train")
    
    print("Fetching Patient Information dataset...")
    # 2. Patient-friendly dataset
    patient_ds = load_dataset("medalpaca/medical_meadow_wikidoc_patient_information", split="train")
    
    return tech_ds, patient_ds


tech_data, patient_data = fetch_medical_meadow()
tech_data.to_json("./train_set/technical_data.jsonl")
patient_data.to_json("./train_set/patient_data.jsonl")

print(f"\nSuccess!")
print(f"Technical samples: {len(tech_data)}")
print(f"Patient samples: {len(patient_data)}")
