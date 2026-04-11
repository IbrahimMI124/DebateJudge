from __future__ import annotations

from .config import LIGHTWEIGHT_MODE, NLI_MODEL_NAME

try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None

# As specified (loaded unless lightweight mode or dependency/model unavailable)
nli_model = None
if not LIGHTWEIGHT_MODE and pipeline is not None:
    try:
        nli_model = pipeline("text-classification", model=NLI_MODEL_NAME)
    except Exception:
        nli_model = None


def _label_is_contradiction(label: str) -> bool:
    # Common forms:
    # - "CONTRADICTION" / "contradiction"
    # - "LABEL_0" (roberta-large-mnli typically maps 0->CONTRADICTION)
    lab = (label or "").upper()
    return lab == "CONTRADICTION" or lab == "LABEL_0"


def compute_speaker_consistency(statements):

    contradictions = 0
    total = 0

    for i in range(len(statements)):
        for j in range(i + 1, len(statements)):

            s1 = statements[i]
            s2 = statements[j]

            if s1["speaker"] != s2["speaker"]:
                continue

            if nli_model is None:
                # Fallback: assume neutral/consistent when model isn't available.
                result_label = "NEUTRAL"
            else:
                # Use proper pair input for MNLI-style models.
                result = nli_model({"text": s1["text"], "text_pair": s2["text"]})
                pred = result[0] if isinstance(result, list) else result
                result_label = pred["label"]

            if _label_is_contradiction(result_label):
                contradictions += 1

            total += 1

    if total == 0:
        return 1.0

    return 1 - (contradictions / total)


def compute_all_consistency(statements):

    speakers = {}

    for stmt in statements:
        speakers.setdefault(stmt["speaker"], []).append(stmt)

    return {sp: compute_speaker_consistency(speakers[sp]) for sp in speakers}
