from .retriever import retrieve
from .judge import judge_claim

def run_verification(claim):
    query = claim["text"]

    evidence = retrieve(claim)
    result = judge_claim(query, evidence)

    result["evidence"] = evidence
    return result