import json
import faiss
from sentence_transformers import SentenceTransformer
import os
import re

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


def _normalize_words(value):
    return re.findall(r"[a-z0-9_]+", (value or "").lower())


def _topic_bonus(attribute, topic):
    if not attribute or not topic:
        return 0.0

    attr = attribute.lower().strip()
    t = topic.lower().strip()

    if attr == t:
        return 1.0

    attr_words = set(_normalize_words(attr))
    topic_words = set(_normalize_words(t))

    if attr_words and topic_words and attr_words.intersection(topic_words):
        return 0.6

    topic_aliases = {
        "individual_awards": {"awards"},
        "goals": {"records", "career_stats", "ucl", "world_cup"},
        "assists": {"assists", "career_stats", "ucl"},
        "ucl": {"ucl"},
        "style": {"style", "legacy"},
        "trophies": {"trophies", "records"},
    }

    if attr in topic_aliases and t in topic_aliases[attr]:
        return 0.5

    return 0.0


def _entity_bonus(entities, text):
    if not entities:
        return 0.0

    lowered_text = text.lower()
    matched = sum(1 for entity in entities if entity and entity.lower() in lowered_text)
    return 1.2 * matched


def _relation_bonus(relation, entities, text):
    if not relation:
        return 0.0

    lowered_text = text.lower()
    relation_cues = {
        "greater_than": ["more than", "most", "higher", "over", "record", "top scorer"],
        "less_than": ["less than", "fewer", "lower"],
        "equal": ["same", "equal", "tied"],
    }

    cues = relation_cues.get(relation.lower(), [])
    cue_match = any(cue in lowered_text for cue in cues)

    if not cue_match:
        return 0.0

    bonus = 0.8
    if entities and len(entities) >= 2:
        first = entities[0].lower() in lowered_text
        second = entities[1].lower() in lowered_text
        if first and second:
            bonus += 0.5

    return bonus


def _extract_query_and_structure(query_or_claim):
    if isinstance(query_or_claim, dict):
        return (
            query_or_claim.get("text", ""),
            query_or_claim.get("entities", []),
            query_or_claim.get("attribute"),
            query_or_claim.get("relation"),
        )

    return str(query_or_claim), [], None, None


def retrieve(query_or_claim, top_k=5):
    query, entities, attribute, relation = _extract_query_and_structure(query_or_claim)
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, 15)

    candidates = []

    for rank, idx in enumerate(indices[0]):
        doc = corpus[idx]
        text = docs[idx]
        topic = doc.get("topic", "")

        semantic_score = -distances[0][rank]
        bonus = keyword_bonus(query, text)
        entity_score = _entity_bonus(entities, text)
        topic_score = _topic_bonus(attribute, topic)
        relation_score = _relation_bonus(relation, entities, text)

        final_score = (
            semantic_score
            + (0.5 * bonus)
            + entity_score
            + topic_score
            + relation_score
        )

        candidates.append((final_score, doc))

    candidates.sort(reverse=True, key=lambda x: x[0])

    return [item[1] for item in candidates[:top_k]]