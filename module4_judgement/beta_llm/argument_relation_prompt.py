from __future__ import annotations


def build_argument_relation_prompt(statement_a: str, statement_b: str) -> str:
    # IMPORTANT: This prompt is intentionally kept as the user provided it.
    return (
        "You are an expert argumentation analyst.\n\n"
        "Task:\n"
        "Given Statement A and Statement B, classify how Statement B relates to Statement A.\n\n"
        "Choose EXACTLY ONE label from:\n\n"
        "Positive:\n\n"
        "* supports = B gives a reason supporting A\n"
        "* evidence_for = B provides data, example, authority, or factual evidence supporting A\n\n"
        "Negative:\n\n"
        "* rebuts = B argues A's conclusion is wrong\n"
        "* contradicts = B states the opposite or incompatible claim\n"
        "* undercuts = B attacks the reasoning link or assumption behind A\n"
        "* counterexample = B gives a case where A fails\n\n"
        "Nuance:\n\n"
        "* qualifies = B limits, narrows, or adds conditions to A\n"
        "* concedes = B accepts part of A before adding something else\n"
        "* clarifies = B explains ambiguity or defines terms in A\n\n"
        "Tie-breaking rules:\n\n"
        "1. If B gives explicit statistics/examples as support -> evidence_for\n"
        "2. If B gives one failing case -> counterexample\n"
        "3. If B attacks hidden assumption -> undercuts\n"
        "4. If B directly says A is false -> contradicts\n"
        "5. If B partially agrees with limits -> qualifies\n"
        "6. If B first agrees then pivots -> concedes\n"
        "7. If B mainly explains wording -> clarifies\n"
        "8. Else if negative disagreement -> rebuts\n"
        "9. Else if positive agreement -> supports\n\n"
        "Return ONLY valid JSON:\n\n"
        "{\n"
        '"label": "...",\n'
        '"confidence": 0.00,\n'
        '"rationale": "max 30 words"\n'
        "}\n\n"
        "Statement A:\n"
        f"{statement_a}\n\n"
        "Statement B:\n"
        f"{statement_b}\n"
    )
