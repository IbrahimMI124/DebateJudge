#!/usr/bin/env python3
"""Quick test of Qwen inference"""
import os
import sys

# Set backend before imports
os.environ["DEBATEJUDGE_NLI_BACKEND"] = "qwen"
sys.path.insert(0, '/mnt/d/6th_sem/NLP_DL/Project/repo/DebateJudge')

print("1. Loading modules...", flush=True)
from module4_judgement.beta_llm.qwen_pair_classifier import classify_argument_relation

print("2. Starting inference test...", flush=True)

# Test 1: Simple statements
print("3. Test 1: Simple statements", flush=True)
result1 = classify_argument_relation(
    "AI is beneficial",
    "AI causes harm"
)
print("Result 1:", result1, flush=True)

# Test 2: Debate statements  
print("4. Test 2: Debate statements", flush=True)
result2 = classify_argument_relation(
    "Market competition and reputation incentivize companies to build safe AI without heavy regulation.",
    "Reputational incentives often fail because harms aren't immediately visible to users."
)
print("Result 2:", result2, flush=True)

print("5. All tests complete!", flush=True)
