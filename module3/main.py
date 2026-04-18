import json
from verifier.pipeline import run_verification

# Simulated claims from Module 2
claims = [
    {
        "id": 1,
        "text": "Messi has more Ballon d'Ors than Ronaldo.",
        "entities": ["Messi", "Ronaldo"],
        "attribute": "individual_awards",
        "relation": "greater_than",
        "claim_type": "statistical"
    },
    {
        "id": 2,
        "text": "Ronaldo scored over 900 career goals, the most in history.",
        "entities": ["Ronaldo"],
        "attribute": "goals",
        "relation": "greater_than",
        "claim_type": "statistical"
    },
    {
        "id": 3,
        "text": "In my opinion, Neymar is more entertaining than Mbappe.",
        "entities": ["Neymar", "Mbappe"],
        "attribute": "style",
        "relation": "greater_than",
        "claim_type": "opinion"
    },
    {
        "id": 4,
        "text": "Neuer won more UCLs than Buffon.",
        "entities": ["Neuer", "Buffon"],
        "attribute": "ucl",
        "relation": "greater_than",
        "claim_type": "statistical"
    }
]

outputs = []

for claim in claims:
    print(f"Processing Claim {claim['id']}...")
    result = run_verification(claim)
    result["id"] = claim["id"]
    outputs.append(result)

print("\nFinal Results:\n")
print(json.dumps(outputs, indent=2))