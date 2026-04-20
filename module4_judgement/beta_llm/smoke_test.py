from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
	p = argparse.ArgumentParser(description="Smoke-test the beta Qwen pair classifier")
	p.add_argument("--backend", default="qwen", help="NLI backend: qwen or mnli")
	p.add_argument("--qwen-model", default=None, help="Local path or HF id for Qwen model")
	p.add_argument("--hf-cache-dir", default=None, help="Optional HF cache dir")
	p.add_argument("--qwen-device-map", default=None, help="Optional device_map (e.g. auto); requires accelerate")
	p.add_argument(
		"--A",
		default="Market competition and reputation incentivize companies to build safe AI without heavy regulation.",
		help="Statement A (premise)",
	)
	p.add_argument(
		"--B",
		default="Reputational incentives often fail because harms aren't immediately visible to users.",
		help="Statement B (hypothesis)",
	)
	args = p.parse_args()

	# Set env vars BEFORE importing the modules that read them.
	os.environ["DEBATEJUDGE_NLI_BACKEND"] = (args.backend or "").strip().lower()
	if args.qwen_model:
		os.environ["DEBATEJUDGE_QWEN_MODEL"] = args.qwen_model
	if args.hf_cache_dir:
		os.environ["DEBATEJUDGE_HF_CACHE_DIR"] = args.hf_cache_dir
	if args.qwen_device_map is not None:
		os.environ["DEBATEJUDGE_QWEN_DEVICE_MAP"] = args.qwen_device_map

	print("python:", sys.executable)
	print("DEBATEJUDGE_NLI_BACKEND=", os.getenv("DEBATEJUDGE_NLI_BACKEND"))
	print("DEBATEJUDGE_QWEN_MODEL=", os.getenv("DEBATEJUDGE_QWEN_MODEL"))
	print("DEBATEJUDGE_QWEN_DEVICE_MAP=", os.getenv("DEBATEJUDGE_QWEN_DEVICE_MAP"))
	print()

	if (args.backend or "").strip().lower().startswith("qwen"):
		from module4_judgement.beta_llm.qwen_pair_classifier import (
			classify_argument_relation,
			map_rich_label_to_nli,
		)

		rich = classify_argument_relation(args.A, args.B)
		print("rich:", rich)
		print("mapped_nli:", map_rich_label_to_nli(str(rich.get("label", ""))))
		print()

	from module4_judgement.nli import classify_pair

	print("classify_pair(A,B):", classify_pair(args.A, args.B))
	print("classify_pair(B,A):", classify_pair(args.B, args.A))


if __name__ == "__main__":
	main()
