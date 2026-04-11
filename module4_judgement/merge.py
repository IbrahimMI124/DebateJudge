from __future__ import annotations


def merge_inputs(statements, claims, facts):
    # MUST: Merge safely and handle missing fields
    combined = {}

    for s in statements:
        combined[s["id"]] = {
            "statement": s,
            "claim": {},
            "fact": {},
        }

    for c in claims:
        if c["id"] in combined:
            combined[c["id"]]["claim"] = c

    for f in facts:
        if f["id"] in combined:
            combined[f["id"]]["fact"] = f

    return combined
