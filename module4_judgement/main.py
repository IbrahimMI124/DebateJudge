from __future__ import annotations

from .aggregation import (
    aggregate_scores,
    apply_consistency_penalty,
    decide_winner,
    normalize_by_statement_count,
)
from .config import normalize_speaker_scores_enabled
from .explanation import generate_explanation
from .merge import merge_inputs
from .nli import compute_all_consistency
from .preprocessing import preprocess_statements
from .scoring import score_all_statements
from .utils import ensure_list


def run_judgement(statements, claims, facts, topic=None):

    statements = preprocess_statements(statements)

    combined = merge_inputs(statements, ensure_list(claims), ensure_list(facts))

    statement_scores = score_all_statements(combined, topic, statements)

    speaker_scores = aggregate_scores(statement_scores, combined)

    if normalize_speaker_scores_enabled():
        speaker_scores = normalize_by_statement_count(speaker_scores, combined)

    consistency = compute_all_consistency(statements)

    final_scores = apply_consistency_penalty(speaker_scores, consistency)

    winner = decide_winner(final_scores)

    explanation = generate_explanation(final_scores, consistency, statement_scores, combined)

    return {
        "speaker_scores": final_scores,
        "consistency": consistency,
        "winner": winner,
        "explanation": explanation,
    }
