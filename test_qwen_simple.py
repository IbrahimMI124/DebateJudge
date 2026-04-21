#!/usr/bin/env python3
import os
os.environ["DEBATEJUDGE_NLI_BACKEND"] = "qwen"

import sys
sys.path.insert(0, '.')

print("Loading classifier...")
from module4_judgement.beta_llm.qwen_pair_classifier import classify_argument_relation

print("Testing with simple statements...")
result = classify_argument_relation(
    "Market competition safe AI",
    "Reputational incentives fail"
)
print("Result:", result)
