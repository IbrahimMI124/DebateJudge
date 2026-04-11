from __future__ import annotations

from .config import get_tie_epsilon, return_tie_enabled


def aggregate_scores(statement_scores, combined):
    speaker_scores = {}

    for id, score in statement_scores.items():
        speaker = combined[id]["statement"]["speaker"]

        speaker_scores.setdefault(speaker, 0)
        speaker_scores[speaker] += score

    return speaker_scores


def apply_consistency_penalty(scores, consistency):
    return {sp: scores[sp] * consistency.get(sp, 1.0) for sp in scores}


def decide_winner(scores):
    if not scores:
        return "Tie" if return_tie_enabled() else None

    # Default behavior (spec) unless tie mode is enabled.
    if not return_tie_enabled():
        return max(scores, key=scores.get)

    # Unbiased tie handling: compare top two scores.
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_speaker, top_score = items[0]

    if len(items) == 1:
        return top_speaker

    _, second_score = items[1]
    if abs(top_score - second_score) < get_tie_epsilon():
        return "Tie"

    return top_speaker


def normalize_by_statement_count(scores, combined):
    """Normalize speaker scores by number of statements per speaker.

    Optional calibration step to reduce advantage from producing many statements.
    """

    counts = {}
    for item in combined.values():
        speaker = item["statement"]["speaker"]
        counts[speaker] = counts.get(speaker, 0) + 1

    normalized = {}
    for speaker, score in scores.items():
        denom = counts.get(speaker, 1)
        normalized[speaker] = score / denom if denom else score

    return normalized
