from __future__ import annotations

from .config import (
    get_scoring_base_bonus,
    get_scoring_evidence_bonus,
    get_scoring_fact_default,
    get_scoring_fact_weight,
    get_scoring_relevance_weight,
    get_time_weight_mild_base,
    get_time_weight_mild_scale,
    get_time_weight_min_denom,
    get_time_weight_mode,
    get_time_weight_none_value,
    get_time_weight_spec_base,
    get_time_weight_spec_scale,
    get_time_weight_total_minus_1_base,
    get_time_weight_total_minus_1_scale,
)
from .relevance import compute_relevance
from .rebuttal import compute_rebuttal_bonus


def _previous_opponent_text(ordered_statements, idx: int) -> str | None:
    """Return the most recent previous statement from a different speaker."""

    if idx <= 0:
        return None

    current_speaker = ordered_statements[idx].get("speaker")
    for j in range(idx - 1, -1, -1):
        if ordered_statements[j].get("speaker") != current_speaker:
            return ordered_statements[j].get("text")
    return None


def compute_time_weight(position, total):
    mode = (get_time_weight_mode() or "spec").lower()
    min_denom = get_time_weight_min_denom()
    min_denom = min_denom if min_denom > 0 else 1.0

    if mode == "none":
        return get_time_weight_none_value()

    if mode == "mild":
        denom = max(total - 1, min_denom)
        base = get_time_weight_mild_base()
        scale = get_time_weight_mild_scale()
        return base + scale * (position / denom)

    if mode == "total_minus_1":
        denom = max(total - 1, min_denom)
        base = get_time_weight_total_minus_1_base()
        scale = get_time_weight_total_minus_1_scale()
        return base + scale * (position / denom)

    # Default: spec behavior
    denom = max(total, min_denom)
    base = get_time_weight_spec_base()
    scale = get_time_weight_spec_scale()
    return base + scale * (position / denom)


def score_statement(item, relevance, position, total):
    fact_score = item.get("fact", {}).get("factual_score", get_scoring_fact_default())
    has_evidence = item.get("claim", {}).get("has_evidence", False)

    fact_w = get_scoring_fact_weight()
    relevance_w = get_scoring_relevance_weight()
    evidence_bonus = get_scoring_evidence_bonus()
    base_bonus = get_scoring_base_bonus()

    score = 0.0

    score += fact_w * fact_score
    score += relevance_w * relevance

    if has_evidence:
        score += evidence_bonus

    score += base_bonus

    score *= compute_time_weight(position, total)

    # Rebuttal awareness (optional): additive bonus if current statement is
    # sufficiently similar to the immediately previous opponent statement.
    # This does not modify the existing base scoring components above.
    rebuttal_info = item.get("_rebuttal", None)
    if isinstance(rebuttal_info, dict):
        score += float(rebuttal_info.get("bonus", 0.0) or 0.0)

    return score


def score_all_statements(combined, topic, ordered_statements):

    scores = {}
    total = len(ordered_statements)

    for idx, stmt in enumerate(ordered_statements):
        item = combined[stmt["id"]]

        relevance = compute_relevance(stmt["text"], topic)

        prev_opp = _previous_opponent_text(ordered_statements, idx)
        item["_rebuttal"] = compute_rebuttal_bonus(stmt["text"], prev_opp)

        score = score_statement(item, relevance, idx, total)

        scores[stmt["id"]] = score

    return scores


def score_statement_detailed(item, relevance, position, total):
    """Return a full breakdown of statement scoring.

    This is an optional helper for debugging/evaluation. It does not change the
    scoring logic used by `score_statement`.
    """

    fact_score = item.get("fact", {}).get("factual_score", get_scoring_fact_default())
    has_evidence = item.get("claim", {}).get("has_evidence", False)

    fact_w = get_scoring_fact_weight()
    relevance_w = get_scoring_relevance_weight()
    evidence_bonus = get_scoring_evidence_bonus()
    base_bonus = get_scoring_base_bonus()

    contrib_fact = fact_w * fact_score
    contrib_relevance = relevance_w * relevance
    contrib_evidence = evidence_bonus if has_evidence else 0.0
    contrib_base = base_bonus

    base_score = contrib_fact + contrib_relevance + contrib_evidence + contrib_base
    weight = compute_time_weight(position, total)

    rebuttal = item.get("_rebuttal") if isinstance(item.get("_rebuttal"), dict) else None
    rebuttal_bonus = float(rebuttal.get("bonus", 0.0) or 0.0) if rebuttal else 0.0

    final_score = base_score * weight + rebuttal_bonus

    return {
        "fact_score": fact_score,
        "relevance": relevance,
        "has_evidence": has_evidence,
        "weights": {
            "fact_weight": fact_w,
            "relevance_weight": relevance_w,
            "evidence_bonus": evidence_bonus,
            "base_bonus": base_bonus,
        },
        "contrib_fact": contrib_fact,
        "contrib_relevance": contrib_relevance,
        "contrib_evidence": contrib_evidence,
        "contrib_base": contrib_base,
        "base_score": base_score,
        "time_weight": weight,
        "rebuttal": rebuttal
        if rebuttal
        else {
            "enabled": False,
            "threshold": None,
            "similarity": None,
            "is_rebuttal": False,
            "bonus": 0.0,
        },
        "final_score": final_score,
    }


def score_all_statements_detailed(combined, topic, ordered_statements):
    """Return per-statement detailed scoring breakdowns keyed by statement id."""

    details = {}
    total = len(ordered_statements)

    for idx, stmt in enumerate(ordered_statements):
        item = combined[stmt["id"]]
        relevance = compute_relevance(stmt["text"], topic)
        prev_opp = _previous_opponent_text(ordered_statements, idx)
        item["_rebuttal"] = compute_rebuttal_bonus(stmt["text"], prev_opp)
        details[stmt["id"]] = score_statement_detailed(item, relevance, idx, total)

    return details
