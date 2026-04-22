# Module 4 — Debate Judgement Engine (module4_judgement)

## Purpose
`module4_judgement` turns a debate (statements + optional claim metadata + optional factuality scores) into:
- final speaker scores
- within-speaker consistency estimates
- a winner (or optional tie)

The public entry point is `module4_judgement.main.run_judgement()`.

## Inputs and Outputs
### Inputs
- `statements`: list of `{id, speaker, text, ...}`
- `claims` (optional): list keyed by `id` (e.g., `{id, has_evidence: bool, ...}`)
- `facts` (optional): list keyed by `id` (e.g., `{id, factual_score: float, ...}`)
- `topic` (optional): string used for relevance scoring

### Output
`run_judgement()` returns:
```json
{
  "speaker_scores": {"A": 1.23, "B": 0.98},
  "consistency": {"A": 1.0, "B": 0.875},
  "winner": "A",
  "explanation": "..."
}
```

## Pipeline Overview
The execution flow in `module4_judgement/main.py` is intentionally simple and modular:
1. **Preprocess**: `preprocess_statements()` sorts statements by `id` (deterministic order).
2. **Merge**: `merge_inputs()` combines `statements`, `claims`, and `facts` by statement `id`.
3. **Score statements**: `score_all_statements()` computes one scalar score per statement.
4. **Aggregate**: `aggregate_scores()` sums statement scores per speaker.
5. **Optional normalization**: `normalize_by_statement_count()` divides by the number of statements per speaker.
6. **Consistency penalty**: `compute_all_consistency()` estimates within-speaker consistency; `apply_consistency_penalty()` multiplies speaker score by consistency.
7. **Winner selection**: `decide_winner()` picks the max score (or optional tie).


## Scoring Mechanism
Statement scoring is implemented in `module4_judgement/scoring.py`.

### Base components
For each statement $s$:
- **Factuality**: a scalar `factual_score` from `facts`
- **Relevance**: cosine similarity between the statement and the debate topic
- **Evidence bonus**: a fixed bonus if `claim.has_evidence` is true

In code (conceptually):

$$
\text{base}(s) = w_f\cdot \text{fact}(s) + w_r\cdot \text{rel}(s) + \mathbf{1}[\text{evidence}]\cdot b_e + b_0
$$

### Time/position weighting
The base score is multiplied by a configurable time weight that increases with position in the debate:

$$
\text{weighted}(s) = \text{base}(s)\cdot \text{time\_weight}(\text{position}, \text{total})
$$

Supported modes are configured via `DEBATEJUDGE_TIME_WEIGHT_MODE` / JSON config:
- `spec` (default)
- `none`
- `mild`
- `total_minus_1`

### Rebuttal bonus
If rebuttal is enabled (`rebuttal.enabled` / `DEBATEJUDGE_REBUTTAL_ENABLED`), an **additive** bonus is applied:
- Find the immediately previous opponent statement.
- Compute embedding cosine similarity to gate whether it is “responding to” that opponent.
- If similarity passes the threshold, run pair classification (default MNLI) from opponent → current.
- If the interaction is a contradiction, add a bonus scaled by similarity.

This is additive by design:

$$
\text{final}(s) = \text{weighted}(s) + \text{rebuttal\_bonus}(s)
$$

Rationale:
- Keeps the core scoring interpretable and stable.
- Allows rebuttal logic to be toggled/calibrated without changing the base components.

### Why this design
The scoring is a weighted, interpretable blend rather than an end-to-end trained judge model:
- **Transparency**: each term has an explicit meaning (factuality, relevance, evidence).
- **Calibratability**: weights and bonuses live in `module4_judgement/config.py` and can be adjusted via env vars or JSON config.
- **Determinism and debuggability**: `score_statement_detailed()` exposes a full breakdown.
- **Graceful degradation**: lightweight mode avoids model downloads and returns conservative fallbacks.

## Consistency Penalty
Consistency is computed in `module4_judgement/nli.py`:
- For each speaker, compare all pairs of that speaker’s statements.
- Count contradictions using pair classification.
- Consistency score is $1 - \frac{\#\text{contradictions}}{\#\text{pairs}}$.

Final speaker score is multiplied by that factor in `apply_consistency_penalty()`.

Rationale:
- Penalizes self-contradiction without requiring a full discourse model.
- Keeps speaker-level reasoning simple: sum first, then apply a single penalty.

## NLI / Pair Classification Backends
`module4_judgement/nli.py` supports a backend switch:
- `mnli` (default): `roberta-large-mnli` via `transformers` pipeline
- `qwen` (beta): local Qwen pair classifier in `module4_judgement/beta_llm/`

The beta Qwen backend uses a richer taxonomy (e.g., supports / rebuts / undercuts / qualifies / …) and maps it back into 3-way NLI labels to preserve the rest of the pipeline.

Configuration:
- Env: `DEBATEJUDGE_NLI_BACKEND=mnli|qwen`
- JSON: `{ "nli": { "backend": "mnli|qwen" } }`

## Configuration and Modes
All tunables are in `module4_judgement/config.py` with precedence:
1. environment variables
2. JSON config pointed to by `DEBATEJUDGE_CONFIG_JSON`
3. defaults

### Lightweight mode
If `DEBATEJUDGE_LIGHTWEIGHT=1`:
- relevance uses a constant fallback
- NLI returns `NEUTRAL`
- rebuttal similarity returns `None` (no bonus)

This allows fast/offline runs while preserving the pipeline shape.

## Limitations
### Heuristic nature of scoring
- The model is not trained end-to-end; weights encode assumptions about what should matter.
- Evidence is a boolean feature (`claim.has_evidence`) rather than verified citation quality.

### Pair classification is imperfect
- MNLI-style NLI is not argumentation-aware; it can miss pragmatic rebuttals and over/under-detect contradiction.
- Within-speaker consistency is a pairwise contradiction count; it does not model topic drift, qualification, or multi-step reasoning.

### LLM backend costs
- The beta `qwen` backend is computationally heavy.
- Running Qwen2.5-7B locally can exceed typical laptop CPU/RAM constraints and may require a capable GPU and significant disk space.

### Multi-class argument-relation taxonomy needs data
- A richer label set (supports / evidence_for / rebuts / undercuts / counterexample / qualifies / concedes / clarifies / contradicts) is useful, but reliable multi-class prediction generally requires:
  - substantial labeled datasets
  - calibration and evaluation across debate domains

## Future Improvements
- **Dataset + training**: build/obtain labeled argument-relation datasets aligned with the taxonomy; train/evaluate a dedicated classifier.
- **Calibration**: learn weights (or per-domain presets) instead of manual tuning; add proper score calibration.
- **Better consistency modeling**: account for qualifications/clarifications rather than treating all non-entailment as neutral.
- **Efficiency**: caching, batching, and optional smaller distilled models for pair classification.
- **Rebuttal structure**: consider more than the immediately previous opponent statement (windowed matching / retrieval).
- **Evidence quality**: replace boolean evidence with evidence span extraction + veracity checks.

## Key Files
- `main.py`: orchestrates the pipeline (`run_judgement`)
- `scoring.py`: per-statement scoring + time weighting
- `relevance.py`: topic relevance via SentenceTransformers
- `rebuttal.py`: similarity-gated rebuttal bonus using pair classification
- `nli.py`: pair classification backend + within-speaker consistency
- `aggregation.py`: speaker aggregation, normalization, tie logic
- `config.py`: all knobs; env/JSON precedence
- `debug_run.py`: CLI/debug harness
- `beta_llm/`: experimental Qwen classifier + prompt utilities
