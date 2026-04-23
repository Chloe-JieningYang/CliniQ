import json

def transform_for_rag_eval(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    transformed = []
    seen_instructions = set()  # deduplicate same questions
    
    for item in data:
        instruction = item.get("instruction", "").strip()
        reference   = item.get("reference", "").strip()
        
        # Skip duplicates (same question appears multiple times with different contexts)
        if instruction in seen_instructions:
            continue
        seen_instructions.add(instruction)
        
        transformed.append({
            "question":  instruction,   # ← feed into RAG
            "reference": reference,     # ← ground truth answer
        })
    
    with open(output_path, 'w') as f:
        json.dump(transformed, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(transformed)} unique questions to {output_path}")

transform_for_rag_eval("mediqa.json", "rag_eval_mediqa.json")