import json
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

def fetch_medical_meadow():
    print("Fetching Technical Wikidoc dataset...")
    # 1. Technical/Professional dataset
    tech_ds = load_dataset("medalpaca/medical_meadow_wikidoc", split="train")
    
    print("Fetching Patient Information dataset...")
    # 2. Patient-friendly dataset
    patient_ds = load_dataset("medalpaca/medical_meadow_wikidoc_patient_information", split="train")
    
    return tech_ds, patient_ds

# --- NEW UTILITY FUNCTIONS ---
clinical_terms = [
    'pathophysiology', 'etiology', 'extensor', 'idiopathic', 
    'sequestration', 'pruritus', 'erythema', 'pathogenesis',
    'granulocytopenia', 'immunoglobulin'
]

def is_too_technical(text):
    """Returns True if the text contains high-level clinical jargon."""
    if not text: return False
    return any(term in text.lower() for term in clinical_terms)

def is_truncated(text):
    """Returns True if the text ends abruptly without proper punctuation."""
    clean_text = text.strip()
    if not clean_text: return True
    return not clean_text.endswith(('.', '!', '?', '"', ')'))
# ------------------------------

def create_dual_tone_dpo(tech_json_path, patient_json_path, output_path, threshold=0.88):
    print("Loading datasets...")
    # Using json.loads for JSONL format
    with open(tech_json_path, 'r') as f:
        tech_data = [json.loads(line) for line in f]
    with open(patient_json_path, 'r') as f:
        patient_data = [json.loads(line) for line in f]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    tech_instructions = [item['input'] for item in tech_data]
    patient_instructions = [item['input'] for item in patient_data]

    print(f"Encoding {len(tech_instructions)} tech and {len(patient_instructions)} patient items...")
    tech_embeddings = model.encode(tech_instructions, convert_to_tensor=True)
    patient_embeddings = model.encode(patient_instructions, convert_to_tensor=True)

    dpo_dataset = []
    skipped_count = 0
    filtered_count = 0

    print(f"Pairing with threshold > {threshold}...")
    for i, p_item in enumerate(patient_data):
        cosine_scores = util.cos_sim(patient_embeddings[i], tech_embeddings)[0]
        best_score = cosine_scores.max().item()
        best_match_idx = cosine_scores.argmax().item()
        
        if best_score >= threshold:
            t_item = tech_data[best_match_idx]
            p_out = p_item['output']
            t_out = t_item['output']

            # --- EXISTING FILTERS ---
            if p_out.strip() == t_out.strip():
                filtered_count += 1
                continue

            placeholders = ["template:", "cme category", "disease name", "risk factor 1"]
            if any(p in t_out.lower() or p in p_out.lower() for p in placeholders):
                filtered_count += 1
                continue

            # --- NEW: TRUNCATION FILTER ---
            if is_truncated(p_out) or is_truncated(t_out):
                filtered_count += 1
                continue

            # --- NEW: PERSONA-AWARE SWAP LOGIC ---
            # We want 'p_out' to be the simple one and 't_out' to be the technical one.
            # If they are reversed in the source, swap them locally before creating pairs.
            if is_too_technical(p_out) and not is_too_technical(t_out):
                p_out, t_out = t_out, p_out

            # --- GENERATE DUAL PAIRS ---
            # Case A: User is a Patient (Simpler is Chosen)
            dpo_dataset.append({
                "prompt": f"I am a patient. {p_item['input']}",
                "chosen": p_out,
                "rejected": t_out
            })

            # Case B: User is a Doctor (Technical is Chosen)
            dpo_dataset.append({
                "prompt": f"I am a doctor. {t_item['input']}",
                "chosen": t_out,
                "rejected": p_out
            })
        else:
            skipped_count += 1

    with open(output_path, 'w') as f:
        for entry in dpo_dataset:
            f.write(json.dumps(entry) + '\n')
    
    print("\n--- Processing Complete ---")
    print(f"Total DPO samples: {len(dpo_dataset)}")
    print(f"Skipped (Similarity): {skipped_count} | Filtered (Quality): {filtered_count}")


tech_data, patient_data = fetch_medical_meadow()
tech_data.to_json("./train_set/technical_data.jsonl")
patient_data.to_json("./train_set/patient_data.jsonl")

print(f"\nSuccess!")
print(f"Technical samples: {len(tech_data)}")
print(f"Patient samples: {len(patient_data)}")

create_dual_tone_dpo('./train_set/technical_data.jsonl', './train_set/patient_data.jsonl', './train_set/dpo_train_data.jsonl', threshold=0.75)