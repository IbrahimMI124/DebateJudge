import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

# Get the directory of the current script (verifier/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the module3 directory
MODULE3_ROOT = os.path.dirname(SCRIPT_DIR)
# Build the full path to the corpus file
CORPUS_PATH = os.path.join(MODULE3_ROOT, "data", "corpus.json")


with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)


docs = [item["text"] for item in corpus]

embeddings = model.encode(docs, convert_to_numpy=True).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def keyword_bonus(query, text):
    score = 0
    q_words = query.lower().split()
    t = text.lower()

    for w in q_words:
        if w in t:
            score += 1

    return score


def retrieve(query, top_k=5):
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, 15)

    candidates = []

    for rank, idx in enumerate(indices[0]):
        text = docs[idx]

        semantic_score = -distances[0][rank]
        bonus = keyword_bonus(query, text)

        final_score = semantic_score + (0.5 * bonus)

        candidates.append((final_score, corpus[idx]))

    candidates.sort(reverse=True, key=lambda x: x[0])

    return [item[1] for item in candidates[:top_k]]