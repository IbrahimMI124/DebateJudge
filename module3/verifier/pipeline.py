from verifier.retriever import retrieve
from verifier.judge import judge_claim

def run_verification(claim):
    query = claim["text"]

    evidence = retrieve(query)
    result = judge_claim(query, evidence)

    result["evidence"] = evidence
    return result