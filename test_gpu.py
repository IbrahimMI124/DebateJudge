#!/usr/bin/env python3
"""Test Qwen with GPU acceleration enabled"""
import os
import sys

# Enable GPU via device_map before imports
os.environ["DEBATEJUDGE_NLI_BACKEND"] = "qwen"
os.environ["DEBATEJUDGE_QWEN_DEVICE_MAP"] = "auto"
sys.path.insert(0, '/mnt/d/6th_sem/NLP_DL/Project/repo/DebateJudge')

print("Testing Qwen with GPU acceleration (device_map=auto)...")
print("Loading classifier...", flush=True)

from module4_judgement.beta_llm.qwen_pair_classifier import classify_argument_relation

print("Running inference...", flush=True)
result = classify_argument_relation(
    "AI is helpful",
    "AI causes problems"
)
print("Result:", result, flush=True)
print("Done!", flush=True)
