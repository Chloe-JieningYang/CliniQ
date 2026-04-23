import sys
import json
from inference import load_model, load_rag_retriever, generate_answer
from retriever import rag_retrieval
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class FaithfulnessEvaluator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"FaithfulnessEvaluator using device: {self.device}")

        print("Loading embedding model...")
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=self.device
        )

        print("Loading NLI model...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        )
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        ).to(self.device)
        self.nli_model.eval()
        print("FaithfulnessEvaluator ready!")

    def lexical_overlap(self, answer, context):
        """Proportion of answer tokens that appear in context."""
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        if len(answer_tokens) == 0:
            return 0.0
        overlap = len(answer_tokens & context_tokens)
        return overlap / len(answer_tokens)

    def embedding_score(self, answer, context):
        """Average max cosine similarity between answer sentences and context sentences."""
        answer_sents = nltk.sent_tokenize(answer)
        context_sents = nltk.sent_tokenize(context)
        if not answer_sents or not context_sents:
            return 0.0
        answer_emb = self.embedder.encode(answer_sents)
        context_emb = self.embedder.encode(context_sents)
        scores = []
        for a_emb in answer_emb:
            sims = cosine_similarity([a_emb], context_emb)[0]
            scores.append(np.max(sims))
        return float(np.mean(scores))

    def entailment_score(self, answer, context):
        """Average NLI entailment probability of each answer sentence given context."""
        answer_sents = nltk.sent_tokenize(answer)
        if not answer_sents:
            return 0.0
        entail_scores = []
        for sent in answer_sents:
            inputs = self.nli_tokenizer(
                context,
                sent,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            with torch.no_grad():
                logits = self.nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
            # label order: [contradiction, neutral, entailment]
            entail_prob = probs[0][2].item()
            entail_scores.append(entail_prob)
        return float(np.mean(entail_scores))

    # def faithfulness_score(self, answer, context):
    #     """Combined faithfulness score: 0.5*entailment + 0.3*embedding + 0.2*lexical."""
    #     lex = self.lexical_overlap(answer, context)
    #     emb = self.embedding_score(answer, context)
    #     ent = self.entailment_score(answer, context)
    #     final_score = (
    #         0.5 * ent +
    #         0.3 * emb +
    #         0.2 * lex
    #     )
    #     return {
    #         "faithfulness": round(final_score, 4),
    #         "entailment":   round(ent, 4),
    #         "embedding":    round(emb, 4),
    #         "lexical":      round(lex, 4),
    #     }
    def faithfulness_score(self, answer, context):
        emb = self.embedding_score(answer, context)
        return {
            "faithfulness": round(emb, 4),
            "embedding":    round(emb, 4),
        }


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    eval_path  = sys.argv[2] if len(sys.argv) > 2 else "rag_eval_mediqa.json"

    # Load model, RAG, and evaluator
    model, tokenizer = load_model(model_path)
    rag_retriever = load_rag_retriever()
    evaluator = FaithfulnessEvaluator()

    # Load eval set
    with open(eval_path) as f:
        eval_set = json.load(f)
    print(f"Loaded {len(eval_set)} questions from {eval_path}")
    
    results = []
    for i, item in enumerate(eval_set, 1):
        question  = item["question"]
        reference = item.get("reference", "")

        print(f"\n[{i}/{len(eval_set)}] Question: {question[:80]}...")

        # Get context from RAG
        context = rag_retrieval(rag_retriever, question)

        # Generate answer WITH RAG
        answer_rag = generate_answer(model, tokenizer, question, context=context)
        scores_rag = evaluator.faithfulness_score(answer_rag, context)
        print(f"  [RAG]    Embedding: {scores_rag['embedding']:.4f}")

        # Generate answer WITHOUT RAG
        answer_no_rag = generate_answer(model, tokenizer, question, context=None)
        scores_no_rag = evaluator.faithfulness_score(answer_no_rag, context)
        print(f"  [No RAG] Embedding: {scores_no_rag['embedding']:.4f}")

        results.append({
            "question":       question,
            "reference":      reference,
            "context":        context,
            "answer_rag":     answer_rag,
            "answer_no_rag":  answer_no_rag,
            "scores_rag":     scores_rag,
            "scores_no_rag":  scores_no_rag,
        })

    # Summary
    avg_rag    = sum(r["scores_rag"]["embedding"]    for r in results) / len(results)
    avg_no_rag = sum(r["scores_no_rag"]["embedding"] for r in results) / len(results)

    print("\n" + "=" * 50)
    print("FAITHFULNESS EVALUATION SUMMARY")
    print("=" * 50)
    print(f"With RAG    : {avg_rag:.4f}")
    print(f"Without RAG : {avg_no_rag:.4f}")
    print(f"Improvement : {avg_rag - avg_no_rag:+.4f}")
    print(f"Total       : {len(results)} questions")
    print("=" * 50)

    # Save results
    with open("faithfulness_results.json", "w") as f:
        json.dump({
            "summary": {
                "faithfulness": round(avg_faith, 4),
                "embedding":    round(avg_emb, 4),
                "total":        len(results),
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print("Full results saved → faithfulness_results.json")

if __name__ == "__main__":
    main()