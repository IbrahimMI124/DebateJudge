from __future__ import annotations

from typing import Any, Dict, Optional

from .config import (
    get_rebuttal_agreement_bonus,
    get_rebuttal_contradiction_bonus,
    get_rebuttal_similarity_threshold,
    get_rebuttal_soft_threshold_band,
    get_rebuttal_soft_threshold_weight,
    rebuttal_enabled,
)
from .nli import classify_pair, label_is_contradiction, label_is_entailment
from .relevance import model as _model
from .relevance import util as _util


def compute_rebuttal_similarity(current_text: str, previous_opponent_text: str) -> Optional[float]:
    """Cosine similarity between current statement and previous opponent statement.

    Uses the same SentenceTransformer instance and cosine similarity util as
    `module4_judgement.relevance`.

    Returns None when embeddings are unavailable (e.g. lightweight mode).
    """

    if _model is None or _util is None:
        return None

    emb1 = _model.encode(current_text, convert_to_tensor=True)
    emb2 = _model.encode(previous_opponent_text, convert_to_tensor=True)
    return float(_util.cos_sim(emb1, emb2))


def compute_rebuttal_bonus(
    current_text: str,
    previous_opponent_text: Optional[str],
) -> Dict[str, Any]:
    """Compute rebuttal bonus against the immediately previous opponent statement.

    Returns a dict with:
      - bonus: float
      - similarity: float|None
      - is_rebuttal: bool
      - threshold: float
      - enabled: bool
    """

    enabled = rebuttal_enabled()
    threshold = get_rebuttal_similarity_threshold()
    soft_band = float(get_rebuttal_soft_threshold_band() or 0.0)
    soft_weight = float(get_rebuttal_soft_threshold_weight() or 0.0)
    soft_weight = max(0.0, min(1.0, soft_weight))

    if not enabled or not previous_opponent_text:
        return {
            "enabled": enabled,
            "threshold": threshold,
            "weight": 0.0,
            "similarity": None,
            "nli_label": None,
            "interaction": "none",
            "is_rebuttal": False,
            "bonus": 0.0,
        }

    similarity = compute_rebuttal_similarity(current_text, previous_opponent_text)
    if similarity is None:
        return {
            "enabled": enabled,
            "threshold": threshold,
            "weight": 0.0,
            "similarity": None,
            "nli_label": None,
            "interaction": "unknown",
            "is_rebuttal": False,
            "bonus": 0.0,
        }

    # Similarity gating with optional soft band.
    # Default `soft_band=0.0` preserves original hard threshold.
    weight = 0.0
    if similarity >= threshold:
        weight = 1.0
    elif soft_band > 0.0 and similarity >= (threshold - soft_band):
        weight = soft_weight

    if weight <= 0.0:
        return {
            "enabled": enabled,
            "threshold": threshold,
            "weight": 0.0,
            "similarity": similarity,
            "nli_label": None,
            "interaction": "below_threshold",
            "is_rebuttal": False,
            "bonus": 0.0,
        }

    nli_label = classify_pair(previous_opponent_text, current_text)

    interaction = "neutral"
    coeff = 0.0
    if label_is_contradiction(nli_label):
        interaction = "contradiction"
        coeff = float(get_rebuttal_contradiction_bonus())
    elif label_is_entailment(nli_label):
        interaction = "agreement"
        coeff = float(get_rebuttal_agreement_bonus())

    bonus = coeff * float(similarity) * float(weight)
    is_rebuttal = interaction == "contradiction" and bonus != 0.0

    return {
        "enabled": enabled,
        "threshold": threshold,
        "weight": weight,
        "similarity": similarity,
        "nli_label": nli_label,
        "interaction": interaction,
        "is_rebuttal": is_rebuttal,
        "bonus": bonus,
    }
