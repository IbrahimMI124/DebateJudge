import re
import json
import spacy

nlp = spacy.load("en_core_web_sm")
#optionally upgrade to en_core_web_trf

# ──────────────────────────────────────────────────────────────────────────────
# NOTE: Argumentative filtering has been intentionally removed from Module 1.
#
# Previously, NON_ARG_PATTERNS / KEEP_OVERRIDES / is_argumentative() lived here.
# That logic was brittle (regex-based) and caused valid claims to be dropped
# before Module 2 ever saw them.
#
# Module 1's only job now is: parse → clean → split into sentences → pass on.
# The binary claim detector in Module 2 (Step 2A) is responsible for deciding
# what is and isn't a claim, using a proper trained classifier.
# ──────────────────────────────────────────────────────────────────────────────

MIN_TOKEN_LENGTH = 3  # Drop only truly empty/trivial fragments (1-2 words)
                      # This is NOT argumentative filtering — just noise removal.


def is_trivial(text: str) -> bool:
    """
    Drop only fragments that carry zero information regardless of context:
      - Single words or very short interjections ("Yeah.", "Okay.", "Hmm.")
      - Completely empty strings after cleaning

    This is intentionally very permissive. A sentence like "That's wrong."
    is only 2 tokens but could be a valid rebuttal — we still keep it.
    We only drop things that are genuinely empty noise.
    """
    tokens = text.split()
    if len(tokens) < MIN_TOKEN_LENGTH:
        return True
    return False


def clean_text(text: str) -> str:
    """
    Light preprocessing only — removes timestamps and normalizes whitespace.
    Does NOT remove filler words (um, uh, like) because:
      - Filler removal can corrupt sentence boundaries for spaCy
      - The binary classifier in Module 2 is robust to informal speech
    """
    # Remove timestamps like [00:01:23] or (00:01)
    text = re.sub(r'\[[\d:]+\]', '', text)
    text = re.sub(r'\([\d:]+\)', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Fix spacing around punctuation
    text = re.sub(r'\s([?.!,])', r'\1', text)

    return text


def normalize_speakers(turns: list) -> list:
    """
    If speakers are named (e.g. "John", "Sarah"), map them to A, B, C...
    Preserves existing A/B labels if already normalized.
    """
    speaker_map = {}
    labels = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    normalized = []
    for turn in turns:
        speaker = turn["speaker"].strip()
        if speaker not in speaker_map:
            speaker_map[speaker] = next(labels)
        normalized.append({
            "speaker": speaker_map[speaker],
            "text": turn["text"]
        })

    return normalized


def parse_plain_text(transcript: str) -> list:
    """
    Handles formats:
      - 'A: some text'
      - '[00:01:23] A: some text'
      - 'John: some text'
      - 'John Smith: some text'
    """
    turns = []
    for line in transcript.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Strip leading timestamps
        line = re.sub(r'^\[[\d:]+\]\s*', '', line)
        line = re.sub(r'^\([\d:]+\)\s*', '', line)

        # Match "Speaker: text" — speaker can be a letter, name, or name with spaces
        match = re.match(r'^([A-Za-z][A-Za-z\s]{0,30}?):\s+(.+)', line)
        if match:
            speaker = match.group(1).strip()
            text    = match.group(2).strip()
            turns.append({"speaker": speaker, "text": text})

    return turns


def parse_json_input(data: dict) -> tuple:
    """
    Handles JSON input:
    {
        "topic": "...",
        "turns": [
            {"speaker": "A", "text": "..."},
            ...
        ]
    }
    """
    topic = data.get("topic", "")
    turns = data.get("turns", [])
    return topic, turns


def process(topic: str, turns: list) -> dict:
    """
    Takes topic string and list of {speaker, text} turns.
    Returns structured JSON ready for Module 2.

    Every sentence that isn't trivially empty is passed through.
    Claim detection is Module 2's responsibility.
    """
    turns = normalize_speakers(turns)

    statements = []
    counter = 1

    for turn in turns:
        speaker = turn["speaker"]
        cleaned = clean_text(turn["text"])

        doc = nlp(cleaned)

        for sent in doc.sents:
            text = sent.text.strip()

            # Only drop genuinely empty/trivial fragments
            if not is_trivial(text):
                statements.append({
                    "id":      counter,
                    "speaker": speaker,
                    "text":    text
                })
                counter += 1

    return {
        "topic":      topic,
        "statements": statements
    }


def run_from_text(topic: str, transcript: str) -> dict:
    """
    Use when input is a plain text transcript string.

    Example:
        topic = "Messi vs Ronaldo: Who is the GOAT?"
        transcript = '''
            A: Messi has won 8 Ballon d'Or awards, more than anyone in history.
            B: Ronaldo has scored over 900 career goals, a record no one else has reached.
        '''
    """
    turns = parse_plain_text(transcript)
    return process(topic, turns)


def run_from_json(data: dict) -> dict:
    """
    Use when input is already a structured JSON dict.

    Example:
        data = {
            "topic": "Messi vs Ronaldo: Who is the GOAT?",
            "turns": [
                {"speaker": "A", "text": "Messi has won 8 Ballon d'Or awards."},
                {"speaker": "B", "text": "Ronaldo scored over 900 career goals."}
            ]
        }
    """
    topic, turns = parse_json_input(data)
    return process(topic, turns)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process debate transcript from file")
    parser.add_argument("--input", required=True, help="Path to transcript file (.txt)")
    parser.add_argument("--topic", required=True, help="Debate topic")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        transcript = f.read()

    result = run_from_text(args.topic, transcript)

    print(json.dumps(result, indent=2))

    print(f"\n✅ Total statements extracted: {len(result['statements'])}")
    print(f"   Speaker A: {sum(1 for s in result['statements'] if s['speaker'] == 'A')}")
    print(f"   Speaker B: {sum(1 for s in result['statements'] if s['speaker'] == 'B')}")
    print(f"\n   NOTE: All non-trivial sentences are passed through.")
    print(f"   Claim detection happens in Module 2 (Step 2A).")