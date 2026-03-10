#!/usr/bin/env python3
"""
Quick test script for the fine-tuned medical Q&A model
"""

import os
import sys
from inference import load_model, generate_answer

# Project root (parent of eval/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(ROOT, "medical_lora_output", "final_model")


def test_model(model_path=None):
    """Test the fine-tuned model with sample medical questions"""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    print("=" * 60)
    print("Loading fine-tuned model...")
    print("=" * 60)

    try:
        model, tokenizer = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure the model has been trained and saved to:")
        print(f"  {model_path}")
        return

    # Test questions
    test_questions = [
        "What are the symptoms of diabetes?",
        "How does aspirin work?",
        "What is hypertension?",
        "Explain the difference between Type 1 and Type 2 diabetes.",
    ]

    print("\n" + "=" * 60)
    print("Testing with sample medical questions")
    print("=" * 60)

    for i, question in enumerate(test_questions, 1):
        print(f"\n[Test {i}/{len(test_questions)}]")
        print(f"Question: {question}")
        print("-" * 60)

        try:
            answer = generate_answer(
                model,
                tokenizer,
                question,
                max_new_tokens=256,
                temperature=0.3,  # Lower temperature for more focused answers
                top_p=0.95
            )
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error generating answer: {e}")

        print("-" * 60)

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    print("\nTo test interactively, run:")
    print("  python eval/inference.py")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    test_model(model_path)
