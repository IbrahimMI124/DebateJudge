# module4_judgement

This module implements the Module 4 Debate Judgement Engine.

For a detailed write-up of the pipeline, scoring rationale, limitations, and future work, see `module4_judgement/MODULE4.md`.

## Where the “result” is
The judgement output is the return value of:

- `module4_judgement.main.run_judgement(statements, claims, facts, topic=None)`

It returns a dict with:
- `speaker_scores`
- `consistency`
- `winner`
- `explanation`

## Lightweight vs full mode
- **Full mode (default):** uses the specified models for relevance + NLI.
- **Lightweight mode:** set `DEBATEJUDGE_LIGHTWEIGHT=1` to avoid downloading models (relevance/NLI fall back).

## NLI backend (MNLI vs Qwen beta)
By default, NLI uses `roberta-large-mnli`.

To switch to the beta local-Qwen pair classifier (rich labels mapped back into 3-way NLI):
- Env var: `DEBATEJUDGE_NLI_BACKEND=qwen`
- JSON config: `{ "nli": { "backend": "qwen" } }`

Important: the system interpreter `/usr/bin/python3` may not have ML dependencies installed.
Use a conda `python` that has the dependencies (e.g., the one already running MNLI), or install them into system Python.

### Smoke test (Qwen)
If Qwen2.5 is already available locally, set `DEBATEJUDGE_QWEN_MODEL` to the local folder.

```bash
cd "/home/mohammed-ibrahim/Downloads/Work/Sem 6/NLPDL/DebateJudge"
unset DEBATEJUDGE_LIGHTWEIGHT

# Use conda python (not /usr/bin/python3)
export DEBATEJUDGE_NLI_BACKEND=qwen
export DEBATEJUDGE_QWEN_MODEL="/ABS/PATH/TO/Qwen2.5-7B-Instruct"

python -m module4_judgement.beta_llm.smoke_test
```

Optional (GPU sharding):

```bash
pip install accelerate
export DEBATEJUDGE_QWEN_DEVICE_MAP=auto
python -m module4_judgement.beta_llm.smoke_test
```

## Calibration flags (optional)
Defaults preserve the original spec behavior.

- `DEBATEJUDGE_TIME_WEIGHT_MODE`: `spec` (default) | `none` | `mild` | `total_minus_1`
- `DEBATEJUDGE_NORMALIZE_SCORES`: `0` (default) | `1`

## Run the debug runner (prints intermediates)
From the project root:

```bash
cd "/home/mohammed-ibrahim/Downloads/Work/Sem 6/NLPDL/DebateJudge"

# Full mode (may download models)
python -m module4_judgement.debug_run --topic "Urban transport policy"

# Lightweight mode (offline/fast)
DEBATEJUDGE_LIGHTWEIGHT=1 python -m module4_judgement.debug_run --topic "Urban transport policy" --lightweight

# Compare weighting modes
DEBATEJUDGE_LIGHTWEIGHT=1 python -m module4_judgement.debug_run --topic "Urban transport policy" --lightweight --time-weight none
DEBATEJUDGE_LIGHTWEIGHT=1 python -m module4_judgement.debug_run --topic "Urban transport policy" --lightweight --time-weight mild

# Enable speaker normalization
DEBATEJUDGE_LIGHTWEIGHT=1 python -m module4_judgement.debug_run --topic "Urban transport policy" --lightweight --normalize
```

A JSON file may also be provided:

```json
{
  "topic": "Urban transport policy",
  "statements": [{"id": 1, "speaker": "A", "text": "..."}],
  "claims": [{"id": 1, "speaker": "A", "has_evidence": true}],
  "facts": [{"id": 1, "factual_score": 1.0}]
}
```

```bash
python -m module4_judgement.debug_run --input-json your_input.json
```

## Sophisticated heavy-mode example
An example debate is included at:
- `module4_judgement/examples/sophisticated_debate.json`

Run it in **heavy mode** (uses `all-MiniLM-L6-v2` + `roberta-large-mnli`; first run may download models):

```bash
cd "/home/mohammed-ibrahim/Downloads/Work/Sem 6/NLPDL/DebateJudge"
unset DEBATEJUDGE_LIGHTWEIGHT
python -m module4_judgement.debug_run --input-json module4_judgement/examples/sophisticated_debate.json
```

Optional fairness/calibration flags:

```bash
# Remove positional bias + allow ties
DEBATEJUDGE_TIME_WEIGHT_MODE=none DEBATEJUDGE_RETURN_TIE=1 \
  python -m module4_judgement.debug_run --input-json module4_judgement/examples/sophisticated_debate.json

# Normalize speaker scores by statement count
DEBATEJUDGE_NORMALIZE_SCORES=1 \
  python -m module4_judgement.debug_run --input-json module4_judgement/examples/sophisticated_debate.json
```
