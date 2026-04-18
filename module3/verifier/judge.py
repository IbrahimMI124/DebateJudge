import ollama
import json

def judge_claim(claim_text, evidence):
    evidence_text = "\n".join(
        [f"- {e['text']}" for e in evidence]
    )

    prompt = f"""
You are a strict football fact-checking judge.

Task:
Evaluate whether the claim is supported by the evidence.

Rules:
- Use ONLY the evidence provided.
- Do NOT invent facts.
- If clearly supported -> score near 1.0
- If contradicted -> score near 0.0
- If partially supported / uncertain -> middle score
- Subjective opinions -> around 0.5

Return ONLY valid JSON:

{{
  "factual_score": float,
  "confidence": float,
  "verdict": "supported / contradicted / uncertain / opinion",
  "reason": "short explanation"
}}

Claim:
{claim_text}

Evidence:
{evidence_text}
"""

    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}]
    )

    text = response["message"]["content"]

    try:
        return json.loads(text)
    except:
        return {
            "factual_score": 0.5,
            "confidence": 0.3,
            "verdict": "uncertain",
            "reason": text
        }