import re
import json
import spacy
 
nlp = spacy.load("en_core_web_sm")

NON_ARG_PATTERNS = [
    r"^(thank you|thanks|hello|hi\b|good (morning|evening|afternoon))",
    r"^(that'?s? (a )?(great|good|fair|interesting|valid) point)",
    r"^(you'?re? (absolutely|completely|totally) right)",
    r"^(moving on|anyway|so,|well,|okay so|alright)",
    r"^(ladies and gentlemen|welcome (to|back))",
    r"^(let'?s? (begin|start|get started|move on))",
    r"^(as i (said|mentioned|stated) (earlier|before|previously))",
    r"^(i (think|believe|feel) (that )?we (can all )?agree)",
]
 
# Phrases that signal a real argument even if they start with "I"
# (opinions and concessions must be preserved for Member 2)
KEEP_OVERRIDES = [
    r"^i (admit|concede|grant|acknowledge)",   # concessions
    r"^i (believe|think|feel|argue|maintain|claim|contend)",  # opinions
    r"^honestly,",
    r"^frankly,",
    r"^let'?s? be real",
    r"^you (said|claimed|stated|argued)",      # contradicts
    r"^if (messi|ronaldo|he|she|they|we)",     # hypotheticals
    r"^had ",                                  # hypotheticals ("Had Ronaldo stayed...")
    r"^sure,",                                 # concessions
    r"^okay,? i'?ll grant",                   # concessions
]

def is_argumentative(sentence: str) -> bool:
    s = sentence.lower().strip()
 
    # Too short to be meaningful
    if len(s.split()) < 7:
        return False
 
    # Check keep overrides first — these are always arguments
    for pattern in KEEP_OVERRIDES:
        if re.search(pattern, s):
            return True
 
    # Check non-argument patterns
    for pattern in NON_ARG_PATTERNS:
        if re.search(pattern, s):
            return False
 
    # Pure questions are usually not claims (rhetorical questions are edge cases)
    # Only discard if the ENTIRE sentence is a question
    if s.endswith("?") and not any(
        kw in s for kw in ["more", "less", "better", "worse", "higher", "lower", "greater"]
    ):
        return False
 
    return True

def clean_text(text: str) -> str:
    # Remove timestamps like [00:01:23] or (00:01)
    text = re.sub(r'\[[\d:]+\]', '', text)
    text = re.sub(r'\([\d:]+\)', '', text)
 
    # Remove filler words (only standalone, not inside words)
    fillers = r'\b(um+|uh+|er+|ah+|hmm+|you know|like,|so,|i mean,)\b'
    text = re.sub(fillers, '', text, flags=re.IGNORECASE)
 
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
    Returns structured JSON ready for Member 2.
    """
    # Normalize speaker names to A, B, C...
    turns = normalize_speakers(turns)
 
    statements = []
    counter = 1
 
    for turn in turns:
        speaker = turn["speaker"]
        cleaned = clean_text(turn["text"])
 
        # Use spaCy to split into individual sentences
        doc = nlp(cleaned)
 
        for sent in doc.sents:
            text = sent.text.strip()
 
            if is_argumentative(text):
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

    # Read transcript file
    with open(args.input, "r", encoding="utf-8") as f:
        transcript = f.read()

    # Run pipeline
    result = run_from_text(args.topic, transcript)

    # Print JSON output
    print(json.dumps(result, indent=2))

    # Stats
    print(f"\n✅ Total argumentative statements extracted: {len(result['statements'])}")
    print(f"   Speaker A: {sum(1 for s in result['statements'] if s['speaker'] == 'A')}")
    print(f"   Speaker B: {sum(1 for s in result['statements'] if s['speaker'] == 'B')}")