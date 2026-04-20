from __future__ import annotations

from .config import NLI_MODEL_NAME, get_nli_backend, is_lightweight_mode


_MNLI_MODEL = None


def _get_mnli_model():
    global _MNLI_MODEL

    if _MNLI_MODEL is not None:
        return _MNLI_MODEL

    if is_lightweight_mode():
        return None

    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import-not-found]
    except Exception:
        return None

    if hf_pipeline is None:
        return None
    try:
        _MNLI_MODEL = hf_pipeline("text-classification", model=NLI_MODEL_NAME)
    except Exception:
        _MNLI_MODEL = None
    return _MNLI_MODEL


def _label_is_contradiction(label: str) -> bool:
    # Common forms:
    # - "CONTRADICTION" / "contradiction"
    # - "LABEL_0" (roberta-large-mnli typically maps 0->CONTRADICTION)
    lab = (label or "").upper()
    return lab == "CONTRADICTION" or lab == "LABEL_0"


def _normalize_nli_label(label: str) -> str:
    """Normalize MNLI labels across model/pipeline variants.

    Common outputs include:
    - "CONTRADICTION" / "NEUTRAL" / "ENTAILMENT"
    - "LABEL_0" / "LABEL_1" / "LABEL_2" (roberta-large-mnli typical mapping)
    """

    lab = (label or "").upper()
    if lab in {"CONTRADICTION", "NEUTRAL", "ENTAILMENT"}:
        return lab
    if lab == "LABEL_0":
        return "CONTRADICTION"
    if lab == "LABEL_1":
        return "NEUTRAL"
    if lab == "LABEL_2":
        return "ENTAILMENT"
    return lab or "NEUTRAL"


def label_is_entailment(label: str) -> bool:
    return _normalize_nli_label(label) == "ENTAILMENT"


def label_is_contradiction(label: str) -> bool:
    return _normalize_nli_label(label) == "CONTRADICTION"


def classify_pair(premise_text: str, hypothesis_text: str) -> str:
    """Classify (premise, hypothesis) as ENTAILMENT/NEUTRAL/CONTRADICTION.

    Returns "NEUTRAL" when the NLI model isn't available.
    """

    if is_lightweight_mode():
        return "NEUTRAL"

    backend = (get_nli_backend() or "").strip().lower()
    if backend.startswith("qwen") or backend in {"llm", "beta_llm"}:
        try:
            from .beta_llm.qwen_pair_classifier import (  # type: ignore
                classify_argument_relation,
                map_rich_label_to_nli,
            )

            rich = classify_argument_relation(premise_text, hypothesis_text)
            return map_rich_label_to_nli(str(rich.get("label", "")))
        except Exception:
            return "NEUTRAL"

    mnli_model = _get_mnli_model()
    if mnli_model is None:
        return "NEUTRAL"

    result = mnli_model({"text": premise_text, "text_pair": hypothesis_text})
    pred = result[0] if isinstance(result, list) else result
    return _normalize_nli_label(pred.get("label"))


def compute_speaker_consistency(statements):

    contradictions = 0
    total = 0

    for i in range(len(statements)):
        for j in range(i + 1, len(statements)):

            s1 = statements[i]
            s2 = statements[j]

            if s1["speaker"] != s2["speaker"]:
                continue

            result_label = classify_pair(s1["text"], s2["text"])

            if label_is_contradiction(result_label):
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
