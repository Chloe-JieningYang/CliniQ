from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

datasets = {
    "medical_meadow_health_advice": "medalpaca/medical_meadow_health_advice",
    "medical_meadow_medical_flashcards": "medalpaca/medical_meadow_medical_flashcards",
    "medical_meadow_medqa": "medalpaca/medical_meadow_medqa",
    "medical_meadow_mmmlu": "medalpaca/medical_meadow_mmmlu",
    "medical_meadow_pubmed_causal": "medalpaca/medical_meadow_pubmed_causal",
    
}

for name, repo_id in datasets.items():
    print(f"Downloading {name}...")
    ds = load_dataset(repo_id, split="train")
    ds.to_json(f"data/{name}.json")
    print(f"  Saved {len(ds)} samples")