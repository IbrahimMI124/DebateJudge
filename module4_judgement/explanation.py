from __future__ import annotations

from .config import get_time_weight_mode, normalize_speaker_scores_enabled
from .aggregation import decide_winner


def _compute_speaker_summary(combined):
    summary = {}
    for item in combined.values():
        stmt = item.get("statement", {})
        speaker = stmt.get("speaker")
        if speaker is None:
            continue

        summary.setdefault(
            speaker,
            {
                "statements": 0,
                "evidence_claims": 0,
                "fact_sum": 0.0,
                "fact_count": 0,
            },
        )

        summary[speaker]["statements"] += 1

        if item.get("claim", {}).get("has_evidence", False):
            summary[speaker]["evidence_claims"] += 1

        if "factual_score" in item.get("fact", {}):
            summary[speaker]["fact_sum"] += float(item["fact"]["factual_score"])
            summary[speaker]["fact_count"] += 1

    for speaker, s in summary.items():
        s["avg_fact"] = (s["fact_sum"] / s["fact_count"]) if s["fact_count"] else None

    return summary


def generate_explanation(scores, consistency, stmt_scores, combined):

    A = scores.get("A", 0)
    B = scores.get("B", 0)

    explanation = f"""
Speaker A Score: {A:.2f}
Speaker B Score: {B:.2f}

Consistency:
A: {consistency.get('A', 1.0):.2f}
B: {consistency.get('B', 1.0):.2f}
"""

    summary = _compute_speaker_summary(combined)
    counts = {sp: s["statements"] for sp, s in summary.items()}

    explanation += f"\n\nStatement counts: A={counts.get('A', 0)}, B={counts.get('B', 0)}"
    explanation += f"\nTime weighting mode: {get_time_weight_mode()}"
    explanation += f"\nNormalized scores: {'yes' if normalize_speaker_scores_enabled() else 'no'}"

    winner = decide_winner(scores) if scores else "Tie"
    delta = abs(A - B)
    reasons = []

    if delta < 0.05:
        reasons.append("very similar overall scores")

    a_cons = consistency.get("A", 1.0)
    b_cons = consistency.get("B", 1.0)
    if a_cons != b_cons:
        reasons.append("difference in consistency")

    a_ev = summary.get("A", {}).get("evidence_claims", 0)
    b_ev = summary.get("B", {}).get("evidence_claims", 0)
    if a_ev != b_ev:
        reasons.append("different amount of evidence-backed claims")

    a_avg_fact = summary.get("A", {}).get("avg_fact")
    b_avg_fact = summary.get("B", {}).get("avg_fact")
    if a_avg_fact is not None and b_avg_fact is not None and abs(a_avg_fact - b_avg_fact) > 0.05:
        reasons.append("difference in average factual scores")

    if counts.get("A", 0) != counts.get("B", 0) and not normalize_speaker_scores_enabled():
        reasons.append("different number of statements (unnormalized sum)")

    if get_time_weight_mode() != "none":
        reasons.append("positional weighting contributed")

    if not reasons:
        reasons.append("combined scoring factors")

    if winner == "Tie":
        explanation += "\n\nResult: Tie. Both speakers have very similar overall scores."
    else:
        explanation += f"\n\nSpeaker {winner} wins due to " + ", ".join(reasons) + "."

    # Add a small numeric summary (truthful, derived from inputs).
    if "A" in summary or "B" in summary:
        explanation += "\n\nSignal summary:"
        if "A" in summary:
            af = summary["A"]["avg_fact"]
            explanation += f"\nA evidence-backed: {a_ev}, avg_fact: {af:.2f}" if af is not None else f"\nA evidence-backed: {a_ev}, avg_fact: n/a"
        if "B" in summary:
            bf = summary["B"]["avg_fact"]
            explanation += f"\nB evidence-backed: {b_ev}, avg_fact: {bf:.2f}" if bf is not None else f"\nB evidence-backed: {b_ev}, avg_fact: n/a"

    return explanation
