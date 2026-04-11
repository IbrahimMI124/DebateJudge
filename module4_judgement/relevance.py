from __future__ import annotations

from .config import LIGHTWEIGHT_MODE, RELEVANCE_MODEL_NAME

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:  # pragma: no cover
    SentenceTransformer = None
    util = None

# As specified (loaded unless lightweight mode or dependency/model unavailable)
model = None
if not LIGHTWEIGHT_MODE and SentenceTransformer is not None:
    try:
        model = SentenceTransformer(RELEVANCE_MODEL_NAME)
    except Exception:
        model = None


def compute_relevance(text, topic):
    if topic is None:
        return 1.0

    if model is None or util is None:
        # Fallback: cannot load embeddings model (offline/tests)
        return 0.5

    emb1 = model.encode(text, convert_to_tensor=True)
    emb2 = model.encode(topic, convert_to_tensor=True)

    return float(util.cos_sim(emb1, emb2))
