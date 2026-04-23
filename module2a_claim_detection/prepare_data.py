"""
prepare_data.py — Step 2A: Binary Claim Detection
==================================================
Pulls FEVER, DailyDialog, and SWDA datasets, balances them,
and writes train.csv / val.csv ready for fine-tuning.

Run:
    python prepare_data.py

Output:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/label_counts.txt   ← sanity check

──────────────────────────────────────────────────────
LABEL CONVENTION:
    1 = claim      (a checkable factual assertion)
    0 = non-claim  (filler, question, opinion, chit-chat)
──────────────────────────────────────────────────────
"""

import os
import random
import pandas as pd
from datasets import load_dataset

random.seed(42)

# ── Output paths ──────────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/raw",       exist_ok=True)

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH   = "data/processed/val.csv"

# ── How many examples to pull from each source ───────────────────────────────
N_FEVER        = 5000   # positive claims
N_DAILYDIALOG  = 2000   # negative: casual conversation
N_SWDA         = 1000   # negative: spoken dialogue acts


# ═════════════════════════════════════════════════════════════════════════════
# 1. POSITIVE EXAMPLES — FEVER dataset
#    fever "claim" field = well-formed factual assertions.
#    Both SUPPORTS and REFUTES rows are real claims.
#    NOT ENOUGH INFO rows are also claims — we include all three.
# ═════════════════════════════════════════════════════════════════════════════

def load_fever_claims(n: int) -> list[dict]:
    print(f"[FEVER] Loading {n} claim examples...")
    dataset = load_dataset("fever", "v1.0", split="train", trust_remote_code=True)

    claims = []
    for row in dataset:
        text = row["claim"].strip()
        if len(text.split()) >= 5:   # drop very short/malformed claims
            claims.append({"text": text, "label": 1})

    sampled = random.sample(claims, min(n, len(claims)))
    print(f"[FEVER] Collected {len(sampled)} claim examples.")
    return sampled


# ═════════════════════════════════════════════════════════════════════════════
# 2a. NEGATIVE EXAMPLES — DailyDialog
#    Contains everyday conversational English.
#    We use ALL utterances because conversational turns are non-claims by nature.
#    We filter out dialog_act == 4 (inform) since those can be claim-like.
#
#    Dialog act codes:
#      1 = dummy  |  2 = inform  |  3 = question  |  4 = directive
#      5 = commissive
#    We keep acts: 1 (dummy/filler), 3 (questions), 4 (directive), 5 (commissive)
#    We skip act 2 (inform) because "inform" utterances can be factual claims.
# ═════════════════════════════════════════════════════════════════════════════

def load_dailydialog_nonclaims(n: int) -> list[dict]:
    print(f"[DailyDialog] Loading {n} non-claim examples...")
    dataset = load_dataset("roskoN/dailydialog", split="train", trust_remote_code=True)

    non_claims = []
    for dialog, acts in zip(dataset["utterances"], dataset["acts"]):
        for utterance, act in zip(dialog, acts):
            text = utterance.strip()
            # Skip "inform" act (act code 2) — too close to factual claims
            if act == 2:
                continue
            if len(text.split()) >= 5:
                non_claims.append({"text": text, "label": 0})

    sampled = random.sample(non_claims, min(n, len(non_claims)))
    print(f"[DailyDialog] Collected {len(sampled)} non-claim examples.")
    return sampled


# ═════════════════════════════════════════════════════════════════════════════
# 2b. NEGATIVE EXAMPLES — SWDA (Switchboard Dialog Act Corpus)
#    Real spoken American English, annotated with dialogue acts.
#    We keep utterances tagged as:
#      - Backchannel (b)        e.g. "uh-huh", "right", "yeah"
#      - Acknowledge (aa)       e.g. "okay", "alright"
#      - Yes/No answers (ny/nn) e.g. "yeah", "no"
#      - Abandoned/Uninterpretable (%) 
#    These are definitionally not claims.
# ═════════════════════════════════════════════════════════════════════════════

NON_CLAIM_SWDA_ACTS = {"b", "aa", "ny", "nn", "%", "x", "bk", "bf", "ba"}

def load_swda_nonclaims(n: int) -> list[dict]:
    print(f"[SWDA] Loading {n} non-claim examples...")
    try:
        dataset = load_dataset("swda", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"[SWDA] Could not load: {e}")
        print("[SWDA] Skipping SWDA — will compensate with more DailyDialog.")
        return []

    non_claims = []
    for row in dataset:
        act  = str(row.get("damsl_act_tag", "")).strip().lower()
        text = str(row.get("text", "")).strip()

        # Clean SWDA-specific markers like disfluency tags
        text = text.replace("-", " ").replace("/", " ").strip()

        if act in NON_CLAIM_SWDA_ACTS and len(text.split()) >= 3:
            non_claims.append({"text": text, "label": 0})

    sampled = random.sample(non_claims, min(n, len(non_claims)))
    print(f"[SWDA] Collected {len(sampled)} non-claim examples.")
    return sampled


# ═════════════════════════════════════════════════════════════════════════════
# 2c. NEGATIVE EXAMPLES — Manual debate-specific sentences
#    These are the most important negatives for your use case.
#    They cover patterns that NO general dataset has:
#      - Meta-debate talk     ("The burden of proof is on you")
#      - Rhetorical moves     ("Let's be real here")
#      - Hedging              ("I think we can all agree")
#      - Conversational filler specific to football debates
#
#    HOW TO EXTEND THIS:
#      Go through your output_module1.json and manually add sentences
#      that are clearly not claims. Aim for 200+ over time.
#      Each one you add improves in-domain accuracy significantly.
# ═════════════════════════════════════════════════════════════════════════════

MANUAL_DEBATE_NONCLAIMS = [
    # ── Meta-debate / burden of proof ────────────────────────────────────────
    "The burden of proof is on you to demonstrate that.",
    "You need to prove your point before I accept it.",
    "That's exactly what you need to show me.",
    "I'm waiting for you to back that up with evidence.",
    "So you're saying the burden's on me here?",
    "The burden's on you to prove that claim.",

    # ── Rhetorical moves / deflection ────────────────────────────────────────
    "Let's be real about what we're talking about here.",
    "Come on, you know that's not the whole picture.",
    "That's a completely different argument though.",
    "You're moving the goalposts now.",
    "That's not even what we're debating.",
    "Okay but that's beside the point.",
    "You're comparing apples and oranges here.",
    "Let's not get distracted by that.",

    # ── Hedging / uncertainty ─────────────────────────────────────────────────
    "I mean it's hard to say definitively.",
    "Yeah I feel like that's a difficult one.",
    "I'm not sure that's entirely accurate.",
    "It depends on how you look at it.",
    "That's subjective at the end of the day.",
    "Well it really comes down to your definition.",
    "I think we need to be careful here.",
    "To be fair, there's merit on both sides.",

    # ── Agreement / acknowledgement ───────────────────────────────────────────
    "Okay yeah I'll give you that one.",
    "Fair point, I can see where you're coming from.",
    "Yeah that's true to an extent.",
    "I don't necessarily disagree with that.",
    "Alright, that's a fair observation.",
    "Sure, I can see your reasoning there.",

    # ── Thinking out loud / filler ────────────────────────────────────────────
    "Let me think about this for a second.",
    "Hmm, trying to think about trophy count.",
    "Where was I going with that.",
    "Let me rephrase what I was saying.",
    "Wait, I lost my train of thought.",
    "Okay so what I'm trying to say is.",

    # ── Football debate specific filler ───────────────────────────────────────
    "Look when you talk about the greatest of all time.",
    "The GOAT debate is always going to be controversial.",
    "People always bring this up but it's complicated.",
    "Every fan has their own opinion on this.",
    "It really depends on what you value in a footballer.",
    "You can't just reduce it to one metric.",
    "That's what makes this debate so interesting.",
    "Everyone has their own criteria for the GOAT.",

    # ── Questions (non-rhetorical) ────────────────────────────────────────────
    "What metric are you even using to measure that?",
    "How do you define success in football though?",
    "Can you give me a specific example of that?",
    "What does that have to do with the argument?",
    "Are we talking club career or international career?",
    "Which season are you referring to exactly?",
]

MANUAL_DEBATE_CLAIMS = [
    # Conversational factual claims
    "Messi has won eight Ballon d'Or awards, more than anyone in history.",
    "Ronaldo has scored over 900 career goals across all competitions.",
    "Messi won the 2022 World Cup with Argentina.",
    "Ronaldo won the Euros with Portugal in 2016.",
    "Ronaldo won league titles in England, Spain, and Italy.",
    "Messi has more assists than Ronaldo over their careers.",
    "Ronaldo is the all-time top scorer in Champions League history.",
    # Hedged but still claims
    "I think Messi is clearly the better player overall.",
    "I mean numbers-wise, Messi is the best there's ever been.",
    "I feel like Ronaldo's goal record speaks for itself.",
    "I'd argue Messi needed Barcelona's system to reach his potential.",
    "I honestly think Ronaldo has been the more complete player.",
    "I wouldn't say Ronaldo was the defining factor in Euro 2004.",
    "I don't think anyone has even come close to Messi as a footballer.",
    # Comparative claims
    "Ronaldo has won leagues in three different countries, Messi has basically won it in one.",
    "Before Ronaldo, Portugal were never a competitive nation internationally.",
    "Portugal struggled to even qualify for major tournaments before Ronaldo.",
    "Ronaldo carried Portugal to their only major international trophy.",
    "Messi has more Ballon d'Or awards than Ronaldo by a significant margin.",
    "Ronaldo has more international goals than any player in history.",
    # Implicit statistical claims
    "Most goals ever scored, most international goals, most Champions League goals.",
    "Won leagues in three different countries, being the best player every single time.",
    "The only success Portugal have had is with Ronaldo as captain.",
    "Messi has pretty much played in the same style of football his entire career.",
    "Whereas Ronaldo adapted to different systems, different leagues, different players.",
    "Messi's record at Barcelona is unmatched by any player at a single club.",
    "Argentina had never won a World Cup in Messi's era before 2022.",
]

def get_manual_claims() -> list[dict]:
    print(f"[Manual] Loading {len(MANUAL_DEBATE_CLAIMS)} hand-labeled debate claim examples.")
    return [{"text": t, "label": 1} for t in MANUAL_DEBATE_CLAIMS]

def get_manual_nonclaims() -> list[dict]:
    print(f"[Manual] Loading {len(MANUAL_DEBATE_NONCLAIMS)} hand-labeled non-claim examples.")
    return [{"text": t, "label": 0} for t in MANUAL_DEBATE_NONCLAIMS]


# ═════════════════════════════════════════════════════════════════════════════
# 3. COMBINE, BALANCE, SPLIT
# ═════════════════════════════════════════════════════════════════════════════

def build_dataset():
    # ── Collect all examples ──────────────────────────────────────────────────
    positives = load_fever_claims(N_FEVER) + get_manual_claims()
    negatives = (
        load_dailydialog_nonclaims(N_DAILYDIALOG)
        + load_swda_nonclaims(N_SWDA)
        + get_manual_nonclaims()
    )

    print(f"\n[Balance] Positives (claims):     {len(positives)}")
    print(f"[Balance] Negatives (non-claims): {len(negatives)}")

    # ── Balance classes ───────────────────────────────────────────────────────
    # We want roughly 50/50. If negatives < positives, trim positives.
    # A slight imbalance (60/40) is okay but avoid anything worse.
    min_count = min(len(positives), len(negatives))
    positives = random.sample(positives, min_count)
    negatives = random.sample(negatives, min_count)

    all_data = positives + negatives
    random.shuffle(all_data)

    print(f"[Balance] Final dataset size:     {len(all_data)} ({min_count} per class)")

    # ── Train / validation split (85% / 15%) ─────────────────────────────────
    split_idx  = int(len(all_data) * 0.85)
    train_data = all_data[:split_idx]
    val_data   = all_data[split_idx:]

    train_df = pd.DataFrame(train_data)
    val_df   = pd.DataFrame(val_data)

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH,   index=False)

    # ── Write label counts for sanity check ──────────────────────────────────
    with open("data/processed/label_counts.txt", "w") as f:
        f.write(f"Training set: {len(train_df)} rows\n")
        f.write(f"  Claims (1):     {(train_df.label == 1).sum()}\n")
        f.write(f"  Non-claims (0): {(train_df.label == 0).sum()}\n\n")
        f.write(f"Validation set: {len(val_df)} rows\n")
        f.write(f"  Claims (1):     {(val_df.label == 1).sum()}\n")
        f.write(f"  Non-claims (0): {(val_df.label == 0).sum()}\n")

    print(f"\n✅ Data saved:")
    print(f"   {TRAIN_PATH}  ({len(train_df)} rows)")
    print(f"   {VAL_PATH}    ({len(val_df)} rows)")
    print(f"   data/processed/label_counts.txt")

    return train_df, val_df


if __name__ == "__main__":
    build_dataset()