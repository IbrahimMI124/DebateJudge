from __future__ import annotations

from .config import get_time_weight_mode
from .relevance import compute_relevance


def compute_time_weight(position, total):
    mode = (get_time_weight_mode() or "spec").lower()

    if mode == "none":
        return 1.0

    if mode == "mild":
        denom = max(total - 1, 1)
        return 1.0 + 0.1 * (position / denom)

    if mode == "total_minus_1":
        denom = max(total - 1, 1)
        return 0.5 + 0.5 * (position / denom)

    # Default: original spec behavior
    return 0.5 + 0.5 * (position / total)


def score_statement(item, relevance, position, total):

    fact_score = item.get("fact", {}).get("factual_score", 0.5)
    has_evidence = item.get("claim", {}).get("has_evidence", False)

    score = 0

    score += 0.4 * fact_score
    score += 0.2 * relevance

    if has_evidence:
        score += 0.2

    score += 0.2

    score *= compute_time_weight(position, total)

    return score


def score_all_statements(combined, topic, ordered_statements):

    scores = {}
    total = len(ordered_statements)

    for idx, stmt in enumerate(ordered_statements):
        item = combined[stmt["id"]]

        relevance = compute_relevance(stmt["text"], topic)

        score = score_statement(item, relevance, idx, total)

        scores[stmt["id"]] = score

    return scores


def score_statement_detailed(item, relevance, position, total):
    """Return a full breakdown of statement scoring.

    This is an optional helper for debugging/evaluation. It does not change the
    scoring logic used by `score_statement`.
    """

    fact_score = item.get("fact", {}).get("factual_score", 0.5)
    has_evidence = item.get("claim", {}).get("has_evidence", False)

    contrib_fact = 0.4 * fact_score
    contrib_relevance = 0.2 * relevance
    contrib_evidence = 0.2 if has_evidence else 0.0
    contrib_base = 0.2

    base_score = contrib_fact + contrib_relevance + contrib_evidence + contrib_base
    weight = compute_time_weight(position, total)
    final_score = base_score * weight

    return {
        "fact_score": fact_score,
        "relevance": relevance,
        "has_evidence": has_evidence,
        "contrib_fact": contrib_fact,
        "contrib_relevance": contrib_relevance,
        "contrib_evidence": contrib_evidence,
        "contrib_base": contrib_base,
        "base_score": base_score,
        "time_weight": weight,
        "final_score": final_score,
    }


def score_all_statements_detailed(combined, topic, ordered_statements):
    """Return per-statement detailed scoring breakdowns keyed by statement id."""

    details = {}
    total = len(ordered_statements)

    for idx, stmt in enumerate(ordered_statements):
        item = combined[stmt["id"]]
        relevance = compute_relevance(stmt["text"], topic)
        details[stmt["id"]] = score_statement_detailed(item, relevance, idx, total)

    return details
