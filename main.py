import json
import os

# Module 1 import
from module1_preprocessing.preprocess import run_from_text as run_module_1_from_text

from module2a_claim_detection.predict import ClaimDetector

# Module 2 import
from module2_claim_extraction.predict import predict as run_claim_extraction_on_statement

# Module 3 import
from module3.verifier.pipeline import run_verification

# Module 4 import
from module4_judgement.main import run_judgement


# --- Wrapper function for Module 2 ---
def run_module_2(statements, claim_detector):
    """
    Runs the new two-step claim extraction process.
    Step 2A: Use ClaimDetector to identify which statements are actual claims.
    Step 2B: Use the original extractor to get details for only the real claims.
    """
    print("Running Step 2A: Claim Detection...")
    # Use the detector to filter statements from Module 1
    filtered_output = claim_detector.filter_statements(statements)
    
    # Get the list of statements that were classified as claims
    claims_to_process = filtered_output["claims"]
    stats = filtered_output["stats"]
    print(f"  -> Detected {stats['claims']} claims out of {stats['total']} statements ({stats['claim_rate'] * 100:.1f}%).")

    print("\nRunning Step 2B: Detailed Claim Extraction...")
    extracted_claims = []
    for claim_stmt in claims_to_process:
        # Now, run the original (more expensive) extractor on the clean data
        claim_data = run_claim_extraction_on_statement(claim_stmt["text"])
        
        # Add the id and speaker back in for the other modules
        claim_data["id"] = claim_stmt["id"]
        claim_data["speaker"] = claim_stmt["speaker"]
        
        extracted_claims.append(claim_data)
    
    print(f"Successfully extracted details for {len(extracted_claims)} claims.")
    with open('output_module2.json', 'w') as f:
        json.dump(extracted_claims, f, indent=4)
    return extracted_claims
# --- Wrapper function for Module 3 ---
def run_module_3(claims):
    """Runs knowledge base verification on a list of claims from Module 2."""
    print("Running Knowledge Base Verification on each claim...")
    verified_facts = []
    total_claims = len(claims)
    for i, claim in enumerate(claims):
        # Add a print statement to show progress
        print(f"  -> Verifying claim {i+1}/{total_claims} (ID: {claim['id']})...")
        result = run_verification(claim)
        result["id"] = claim["id"]
        verified_facts.append(result)

    print(f"Successfully verified {len(verified_facts)} claims.")
    with open('output_module3.json', 'w') as f:
        json.dump(verified_facts, f, indent=4)
    return verified_facts

# --- Main Pipeline Function ---
def run_debate_judge_pipeline(debate_transcript_path, topic):
    """Runs the full NLP-based debate judge pipeline."""
    print("Starting Debate Judge Pipeline...")

    # --- Module 1: Argument Segmentation ---
    print("\n[Module 1] Running Argument Segmentation...")
    with open(debate_transcript_path, 'r') as f:
        raw_text = f.read()
    module1_output = run_module_1_from_text(topic=topic, transcript=raw_text)
    statements = module1_output.get("statements", [])
    with open('output_module1.json', 'w') as f:
        json.dump(statements, f, indent=4)
    print(f"[Module 1] Success: Extracted {len(statements)} argumentative statements.")

    # --- Module 2: Claim Extraction ---
    print("\n[Module 2] Pre-loading model (if needed) and running extraction...")
    claim_detector = ClaimDetector()
    claims = run_module_2(statements, claim_detector)

    # --- Module 3: Knowledge Base & Verification ---
    print("\n[Module 3] Pre-loading knowledge base (if needed) and running verification...")
    facts = run_module_3(claims)

    # --- Module 4: NLI + Scoring & Winner Decision ---
    print("\n[Module 4] Running Judgement...")
    if os.environ.get("DEBATEJUDGE_LIGHTWEIGHT"):
        del os.environ["DEBATEJUDGE_LIGHTWEIGHT"]
    final_result = run_judgement(statements=statements, claims=claims, facts=facts, topic=topic)

    print("\n--- Pipeline Finished ---")
    print("\nFinal Result:")
    print(json.dumps(final_result, indent=4))
    
    with open('output_final_judgement.json', 'w') as f:
        json.dump(final_result, f, indent=4)

    return final_result

if __name__ == '__main__':
    DEBATE_FILE = 'data/debate1.txt' # Make sure this file exists
    DEBATE_TOPIC = "Messi vs Ronaldo"
    run_debate_judge_pipeline(DEBATE_FILE, DEBATE_TOPIC)