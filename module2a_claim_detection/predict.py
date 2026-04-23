"""
predict.py — Step 2A: Binary Claim Detection (Inference)
=========================================================
This is the file that Module 2 actually imports and calls at runtime.
It loads the trained model once, then classifies statements one by one
(or in batches).

Usage as a script (for testing):
    python predict.py --input output_module1.json --output output_module2a.json

Usage as a module (imported by Module 2):
    from predict import ClaimDetector
    detector = ClaimDetector()
    result = detector.predict("Messi has more Ballon d'Or awards than Ronaldo.")
    # → {"text": "...", "is_claim": True, "confidence": 0.97}
"""

import json
import argparse
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model", "claim_detector")
MAX_LEN    = 128
BATCH_SIZE = 32                       # For batch prediction — increase if you have GPU

# ── Confidence threshold ──────────────────────────────────────────────────────
# Statements where the model confidence is below this threshold are treated
# as non-claims even if the model leans toward "claim".
# 
# Start with 0.65 — this is deliberately conservative.
# If you're getting too many false positives (non-claims getting through),
# raise it to 0.70 or 0.75.
# If you're dropping too many real claims, lower it to 0.60.
CONFIDENCE_THRESHOLD = 0.5


class ClaimDetector:
    """
    Binary claim classifier. Load once, call many times.

    Example:
        detector = ClaimDetector()

        # Single prediction
        result = detector.predict("Ronaldo scored 700 career goals.")
        print(result)
        # {
        #   "text": "Ronaldo scored 700 career goals.",
        #   "is_claim": True,
        #   "confidence": 0.97,
        #   "label": 1
        # }

        # Batch prediction (faster for many statements)
        results = detector.predict_batch([
            "Messi has 8 Ballon d'Or awards.",
            "Yeah I mean it's hard to say.",
            "Ronaldo has scored more Champions League goals."
        ])
    """

    def __init__(self, model_dir: str = MODEL_DIR, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[ClaimDetector] Loading model from: {model_dir}")
        print(f"[ClaimDetector] Device: {self.device}")
        print(f"[ClaimDetector] Confidence threshold: {threshold}")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        print("[ClaimDetector] Ready.\n")

    def predict(self, text: str) -> dict:
        """
        Classify a single statement.

        Returns:
            {
                "text":       str,
                "is_claim":   bool,
                "confidence": float,  ← probability of the predicted class
                "label":      int     ← 1 = claim, 0 = non-claim
            }
        """
        encoding = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        probs      = torch.softmax(outputs.logits, dim=1).squeeze()
        claim_prob = probs[1].item()        # probability of class 1 (claim)
        non_claim_prob = probs[0].item()    # probability of class 0 (non-claim)

        # Apply threshold: only call it a claim if confidence exceeds threshold
        is_claim   = claim_prob >= self.threshold
        confidence = claim_prob if is_claim else non_claim_prob

        return {
            "text":       text,
            "is_claim":   is_claim,
            "confidence": round(confidence, 4),
            "label":      1 if is_claim else 0,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Classify a list of statements. More efficient than calling predict()
        in a loop because it batches tokenization and forward passes.
        """
        results = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]

            encoding = self.tokenizer(
                batch_texts,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids      = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=1)

            for text, prob_row in zip(batch_texts, probs):
                claim_prob     = prob_row[1].item()
                non_claim_prob = prob_row[0].item()
                is_claim       = claim_prob >= self.threshold
                confidence     = claim_prob if is_claim else non_claim_prob

                results.append({
                    "text":       text,
                    "is_claim":   is_claim,
                    "confidence": round(confidence, 4),
                    "label":      1 if is_claim else 0,
                })

        return results

    def filter_statements(self, statements: list[dict]) -> dict:
        """
        Takes the full output from Module 1 (list of statement objects)
        and returns two lists:
            - claims:      statements classified as claims → pass to Step 2B
            - non_claims:  everything else → archived, never reaches Module 3

        Input format (Module 1 output):
            [
                {"id": 1, "speaker": "A", "text": "Messi has 8 Ballon d'Or awards."},
                {"id": 2, "speaker": "B", "text": "Yeah I mean it's hard to say."},
                ...
            ]

        Output format:
            {
                "claims": [
                    {
                        "id": 1, "speaker": "A",
                        "text": "Messi has 8 Ballon d'Or awards.",
                        "is_claim": True, "confidence": 0.97, "label": 1
                    },
                    ...
                ],
                "non_claims": [
                    {
                        "id": 2, "speaker": "B",
                        "text": "Yeah I mean it's hard to say.",
                        "is_claim": False, "confidence": 0.89, "label": 0
                    },
                    ...
                ],
                "stats": {
                    "total": 10,
                    "claims": 6,
                    "non_claims": 4,
                    "claim_rate": 0.6
                }
            }
        """
        texts = [s["text"] for s in statements]
        predictions = self.predict_batch(texts)

        claims, non_claims = [], []

        for statement, prediction in zip(statements, predictions):
            enriched = {**statement, **prediction}
            if prediction["is_claim"]:
                claims.append(enriched)
            else:
                non_claims.append(enriched)

        total = len(statements)
        return {
            "claims":     claims,
            "non_claims": non_claims,
            "stats": {
                "total":      total,
                "claims":     len(claims),
                "non_claims": len(non_claims),
                "claim_rate": round(len(claims) / total, 2) if total > 0 else 0,
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
# CLI — for testing predict.py standalone
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run claim detection on Module 1 output")
    parser.add_argument("--input",     required=True, help="Path to output_module1.json")
    parser.add_argument("--output",    default="output_module2a.json", help="Where to save results")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    args = parser.parse_args()

    # Load Module 1 output
    with open(args.input, "r", encoding="utf-8") as f:
        module1_output = json.load(f)

    if isinstance(module1_output, list):
        topic      = ""
        statements = module1_output
    else:
        topic      = module1_output.get("topic", "")
        statements = module1_output.get("statements", [])

    print(f"[Input] Topic: {topic}")
    print(f"[Input] Statements to classify: {len(statements)}\n")

    # Run claim detection
    detector = ClaimDetector(threshold=args.threshold)
    result   = detector.filter_statements(statements)

    # Save output
    output = {
        "topic":      topic,
        "claims":     result["claims"],
        "non_claims": result["non_claims"],
        "stats":      result["stats"],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    stats = result["stats"]
    print(f"✅ Claim detection complete.")
    print(f"   Total statements:  {stats['total']}")
    print(f"   Claims detected:   {stats['claims']}  → passed to Step 2B")
    print(f"   Non-claims:        {stats['non_claims']}  → archived")
    print(f"   Claim rate:        {stats['claim_rate'] * 100:.1f}%")
    print(f"\n   Output saved to: {args.output}")

    # Print a sample for quick visual check
    print(f"\n── Sample Claims ──────────────────────────────────────")
    for c in result["claims"][:3]:
        print(f"  [{c['confidence']:.2f}] {c['text']}")

    print(f"\n── Sample Non-Claims ──────────────────────────────────")
    for nc in result["non_claims"][:3]:
        print(f"  [{nc['confidence']:.2f}] {nc['text']}")


if __name__ == "__main__":
    main()