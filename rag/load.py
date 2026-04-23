# Run scripts per case!
# Chunk

import json

def convert_cord19_to_txt(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        count = 0
        for entry in data:
            text_input = entry.get("input", "").strip()
            
            if text_input:
                clean_line = " ".join(text_input.split())
                outfile.write(clean_line + "\n")
                count += 1

    print(f"Processed {count} records and saved to {output_file}")

def convert_health_advice(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        # JSONL format - one JSON object per line
        data = [json.loads(line) for line in f if line.strip()]
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        count = 0
        for entry in data:
            text_input = entry.get("input", "").strip()
            if text_input:
                clean_line = " ".join(text_input.split())
                outfile.write(clean_line + "\n")
                count += 1
    print(f"Processed {count} records and saved to {output_file}")

if __name__ == "__main__":
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent
    raw_json_dir = base_dir / "data"
    documents_dir = base_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    convert_health_advice(raw_json_dir / 'medical_meadow_health_advice.json', documents_dir / 'medical_meadow_health_advice.txt')
    convert_health_advice(raw_json_dir / 'medical_meadow_medical_flashcards.json', documents_dir / 'medical_meadow_medical_flashcards.txt')
    convert_health_advice(raw_json_dir / 'medical_meadow_medqa.json', documents_dir / 'medical_meadow_medqa.txt')
    convert_health_advice(raw_json_dir / 'medical_meadow_mmmlu.json', documents_dir / 'medical_meadow_mmmlu.txt')
    convert_health_advice(raw_json_dir / 'medical_meadow_pubmed_causal.json', documents_dir / 'medical_meadow_pubmed_causal.txt')