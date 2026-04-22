# DebateJudge Pipeline (Modules 1–4)

## Purpose
DebateJudge is a modular pipeline that:
1. segments a debate transcript into argumentative statements,
2. extracts structured claims from each statement,
3. verifies those claims against a small knowledge corpus, and
4. scores speakers to produce a final judgement (winner + explanation).

The orchestrator is `main.py` at the repository root.

## Data Contracts (Artifacts)
The pipeline communicates between modules using simple JSON-like Python dicts/lists.

### Module 1 output: statements
A list of:
- `id`: integer, increasing
- `speaker`: normalized label (A, B, C, …)
- `text`: one argumentative sentence

Written to `output_module1.json` by `main.py`.

### Module 2 output: claims
For each statement `id`, Module 2 adds extracted claim metadata:
- `entities`: list of entity strings
- `attribute`, `relation`, `claim_type`, `stance`

Written to `output_module2.json`.

### Module 3 output: facts
For each statement `id`, Module 3 provides a factuality judgement:
- `factual_score`: float in 
  - near `1.0` for supported,
  - near `0.0` for contradicted,
  - around `0.5` for uncertain/opinion,
- `confidence`: float,
- `verdict`: one of `supported | contradicted | uncertain | opinion`,
- `reason`: short explanation or raw model output if parsing fails,
- `evidence`: list of retrieved corpus passages.

Written to `output_module3.json`.

### Module 4 output: final judgement
A single dict with:
- `speaker_scores`: aggregated per-speaker score
- `consistency`: within-speaker consistency estimate
- `winner`: speaker label (or optional tie depending on Module 4 config)
- `explanation`: natural language explanation

Written to `output_final_judgement.json`.

## End-to-End Flow
The pipeline in `main.py` runs:

1. **Module 1 — Preprocessing**
   - Parses a transcript (`Speaker: text` lines).
   - Cleans filler/timestamps.
   - Uses spaCy sentence splitting and heuristics to keep “argumentative” sentences.
   - Normalizes arbitrary speaker names to `A`, `B`, `C`, …

2. **Module 2 — Claim Extraction**
   - Runs a multitask Hugging Face model per statement.
   - Extracts entities (token-tag decoding) and classifies attribute/relation/claim_type/stance.

3. **Module 3 — Verification (RAG)**
   - Retrieves candidate evidence passages from `module3/data/corpus.json` using SentenceTransformers + FAISS.
   - Prompts a local Ollama model (`phi3`) to emit strict JSON with a factual score.
   - Falls back to `{factual_score: 0.5, verdict: "uncertain"}` when JSON parsing fails.

4. **Module 4 — Judgement**
   - Merges `statements` + `claims` + `facts` on `id`.
   - Scores each statement using a weighted blend of factuality, topic relevance, and evidence bonus (plus optional rebuttal bonus).
   - Aggregates statement scores per speaker and applies a within-speaker contradiction penalty.
   - Selects a winner and generates an explanation.

For Module 4 details (including config knobs and the MNLI/Qwen backend switch), see `module4_judgement/MODULE4.md`.

## Design Justifications (Pipeline-Level)
- **Modularity and inspectability**: each stage writes a standalone artifact (`output_module*.json`) that can be inspected and re-used.
- **Separation of concerns**:
  - Module 1 focuses on segmentation/cleanup,
  - Module 2 on structured extraction,
  - Module 3 on evidence-grounded verification,
  - Module 4 on scoring/aggregation and debate-level reasoning.
- **Graceful degradation**: when a stage is uncertain (e.g., verifier JSON parsing), the pipeline uses conservative defaults rather than failing hard.

## Limitations (Pipeline-Level)
- **Domain coupling**: Module 3’s verifier prompt is football-specific (“football fact-checking judge”), which biases behavior outside that domain.
- **Small, synthetic knowledge base**: the corpus is limited and may not cover many claims; retrieval quality dominates verifier quality.
- **Module 2/3 alignment**: claim metadata from Module 2 is not currently used to condition retrieval/judging in Module 3 (the verifier uses only `claim["text"]`).
- **Latency and resource usage**:
  - Module 2 loads a large model and runs per-statement.
  - Module 3 builds embeddings + FAISS index at import time.
  - Module 4 can load multiple models (SentenceTransformers + NLI backend).
- **Heuristic winner computation**: Module 4 is an interpretable scoring system, not a trained end-to-end judge; results reflect configured assumptions.

## Detailed Module Docs
- Module 1: `module1_preprocessing/MODULE1.md`
- Module 2: `module2_claim_extraction/MODULE2.md`
- Module 3: `module3/MODULE3.md`
- Module 4: `module4_judgement/MODULE4.md`
