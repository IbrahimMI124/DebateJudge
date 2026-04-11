# module4_judgement

This module implements the Module 4 Debate Judgement Engine.

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

You can also pass a JSON file:

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
