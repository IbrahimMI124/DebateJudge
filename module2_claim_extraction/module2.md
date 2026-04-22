# Claim Extraction Module

## Overview

This module is designed to extract structured claims from raw text. Given a sentence, the system identifies:

- Entities involved in the claim  
- Attribute being discussed  
- Relation between entities  
- Claim Type (e.g., statistical, factual, opinion)  
- Stance (support / attack / neutral)

This transforms unstructured natural language into a machine-interpretable claim representation, enabling tasks like fact verification and debate scoring.

---

## Model Architecture

The system uses a multi-task learning architecture built on top of a pretrained transformer encoder.

### Base Model

At its core, the model uses:

- `MODEL_NAME = "microsoft/deberta-v3-base"`

This corresponds to a pretrained DeBERTa-v3 model from Hugging Face, which provides strong contextual embeddings and improved attention mechanisms through disentangled attention. The encoder is instantiated via:

```python
AutoModel.from_pretrained(config["model_name"])
```

DeBERTa-v3 is chosen for its superior performance on NLU tasks, especially those requiring fine-grained semantic understanding such as entity extraction and relational reasoning.

---

### Key Components

#### 1. Transformer Encoder
- Converts input text into contextual token embeddings  
- Captures semantic relationships between words using self-attention  

---

#### 2. Contextual Attention Pooling
```python
ContextualAttentionPooling
```

- Learns to focus on the most important tokens in a sentence  
- Produces a single pooled representation of the claim  

Intuition:  
Instead of averaging all tokens, the model learns what matters most (e.g., entities, comparison phrases like "more than")

---

#### 3. NER Head (Token-level Task)

```python
self.ner_linear
```

- Performs Named Entity Recognition  
- Uses BIO tagging (`B-`, `I-`, `O`) to extract entities  

---

#### 4. Multi-Task Classification Heads

Each of the following is predicted from the pooled representation:

- Attribute Head → What is being compared (e.g., "goals", "saves")  
- Relation Head → Type of relation (e.g., greater_than)  
- Claim Type Head → Nature of claim (e.g., statistical)  
- Stance Head → Argument direction (support / attack)  

Each head uses:

```python
Linear → LayerNorm → GELU → Dropout → Linear
```

---

## Inference Pipeline

### Step-by-step Flow

1. Tokenization  
   - Input text is tokenized using a Hugging Face tokenizer  

2. Encoding  
   - Tokens are passed through the transformer to obtain contextual embeddings  

3. NER Prediction  
   - Token-level classification produces entity tags  

4. Entity Decoding  
   - BIO tags are merged into full entity spans  

5. Global Predictions  
   - The pooled representation is used to predict attribute, relation, claim type, and stance  

---

### Example

#### Input:

```text
"Neuer has more saves than Buffon"
```

#### Output:

```json
{
  "text": "Neuer has more saves than Buffon",
  "entities": ["Neuer", "Buffon"],
  "attribute": "saves",
  "relation": "greater_than",
  "claim_type": "statistical",
  "stance": "attack"
}
```

---

## Training Data

The model is trained on a synthetic dataset specifically designed for structured claim extraction.

### Why Synthetic Data?

- Real-world annotated claim datasets are scarce and expensive  
- Debate-style structured claims require fine-grained and consistent labels  
- Synthetic generation enables:
  - Controlled diversity  
  - Balanced class distributions  
  - Scalable data generation  

---

## Design Philosophy

### 1. Structured Understanding

Instead of treating the problem as simple classification, the system decomposes claims into interpretable components.

---

### 2. Multi-Task Learning

All tasks (NER + classification) are learned jointly, allowing shared representations to improve overall performance.

---

### 3. Debate-Oriented Modeling

The system captures not only what is being said, but also:

- How it is said (claim type)  
- Why it is said (stance)  

---
