#!/usr/bin/env python3
"""Quick test script for beta LLM"""
import subprocess
import sys

# Try to run the smoke test
result = subprocess.run(
    [sys.executable, "-m", "module4_judgement.beta_llm.smoke_test", "--backend", "qwen"],
    capture_output=False,
    text=True
)

sys.exit(result.returncode)
