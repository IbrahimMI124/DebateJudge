from __future__ import annotations


def preprocess_statements(statements):
    # MUST: Sort statements by id and preserve order permanently
    return sorted(statements, key=lambda x: x["id"])
