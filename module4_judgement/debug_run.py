from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as: `python module4_judgement/debug_run.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)


def run_and_print(
    statements: List[Dict[str, Any]],
    claims: Optional[List[Dict[str, Any]]] = None,
    facts: Optional[List[Dict[str, Any]]] = None,
    topic: Optional[str] = None,
) -> None:
    # Import after env vars are configured so config-driven behavior applies.
    from module4_judgement.aggregation import (
        aggregate_scores,
        apply_consistency_penalty,
        decide_winner,
        normalize_by_statement_count,
    )
    from module4_judgement.config import normalize_speaker_scores_enabled
    from module4_judgement.explanation import generate_explanation
    from module4_judgement.merge import merge_inputs
    from module4_judgement.nli import compute_all_consistency
    from module4_judgement.preprocessing import preprocess_statements
    from module4_judgement.scoring import score_all_statements_detailed
    from module4_judgement.utils import ensure_list

    statements = preprocess_statements(statements)
    combined = merge_inputs(statements, ensure_list(claims), ensure_list(facts))

    # Compute detailed scoring once and derive `statement_scores` from it.
    # This avoids recomputing embedding-based features twice in the debug run.
    statement_details = score_all_statements_detailed(combined, topic, statements)
    statement_scores = {sid: float(d["final_score"]) for sid, d in statement_details.items()}

    speaker_scores_raw = aggregate_scores(statement_scores, combined)
    speaker_scores = (
        normalize_by_statement_count(speaker_scores_raw, combined)
        if normalize_speaker_scores_enabled()
        else speaker_scores_raw
    )
    consistency = compute_all_consistency(statements)
    final_scores = apply_consistency_penalty(speaker_scores, consistency)
    winner = decide_winner(final_scores)
    explanation = generate_explanation(final_scores, consistency, statement_scores, combined)

    print("=== Per-statement breakdown ===")
    for stmt in statements:
        sid = stmt["id"]
        detail = statement_details[sid]
        print(f"\nID {sid} | Speaker {stmt['speaker']}")
        print(f"Text: {stmt['text']}")
        print(
            f"fact={detail['fact_score']:.3f}  rel={detail['relevance']:.3f}  "
            f"evidence={detail['has_evidence']}  weight={detail['time_weight']:.3f}"
        )
        rebut = detail.get("rebuttal") or {}
        if rebut.get("enabled"):
            sim = rebut.get("similarity")
            sim_str = f"{sim:.3f}" if isinstance(sim, (int, float)) else "n/a"
            print(
                f"rebuttal: interaction={rebut.get('interaction')}  label={rebut.get('nli_label')}  "
                f"is_rebuttal={rebut.get('is_rebuttal')}  sim={sim_str}  "
                f"thr={rebut.get('threshold')}  w={rebut.get('weight')}  "
                f"bonus={rebut.get('bonus')}"
            )
        print(
            f"base={detail['base_score']:.3f} -> final={detail['final_score']:.3f}  "
            f"(fact {detail['contrib_fact']:.3f} + rel {detail['contrib_relevance']:.3f} + "
            f"ev {detail['contrib_evidence']:.3f} + base {detail['contrib_base']:.3f})"
        )

    print("\n=== Aggregation ===")
    print("statement_scores:")
    print(_pretty(statement_scores))
    print("speaker_scores_raw (pre-penalty):")
    print(_pretty(speaker_scores_raw))
    if speaker_scores != speaker_scores_raw:
        print("speaker_scores_normalized (pre-penalty):")
        print(_pretty(speaker_scores))
    print("consistency:")
    print(_pretty(consistency))
    print("final_scores:")
    print(_pretty(final_scores))

    print("\n=== Winner ===")
    print(winner)

    print("\n=== Explanation ===")
    print(explanation.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug runner for module4_judgement")
    parser.add_argument("--topic", default=None)
    parser.add_argument(
        "--config-json",
        default=None,
        help="Path to a DebateJudge JSON config file (sets DEBATEJUDGE_CONFIG_JSON)",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Set DEBATEJUDGE_LIGHTWEIGHT=1 for offline/fast runs",
    )
    parser.add_argument(
        "--time-weight",
        default=None,
        choices=["spec", "none", "mild", "total_minus_1"],
        help="Override DEBATEJUDGE_TIME_WEIGHT_MODE for this run",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Set DEBATEJUDGE_NORMALIZE_SCORES=1 for this run",
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help="Optional path to a JSON file with keys: statements, claims, facts, topic",
    )

    args = parser.parse_args()

    if args.config_json:
        os.environ["DEBATEJUDGE_CONFIG_JSON"] = args.config_json

    if args.lightweight:
        os.environ["DEBATEJUDGE_LIGHTWEIGHT"] = "1"

    if args.time_weight:
        os.environ["DEBATEJUDGE_TIME_WEIGHT_MODE"] = args.time_weight

    if args.normalize:
        os.environ["DEBATEJUDGE_NORMALIZE_SCORES"] = "1"

    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        run_and_print(
            statements=payload["statements"],
            claims=payload.get("claims"),
            facts=payload.get("facts"),
            topic=payload.get("topic", args.topic),
        )
        return

    # Default demo scenarios
    run_and_print(
        statements=[
            {"id": 1, "speaker": "A", "text": "Cars cause pollution and traffic."},
            {"id": 2, "speaker": "B", "text": "Cars are necessary for economic growth."},
        ],
        claims=[],
        facts=[],
        topic=args.topic or "Urban transport policy",
    )


if __name__ == "__main__":
    main()
