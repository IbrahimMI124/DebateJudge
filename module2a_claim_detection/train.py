"""
train.py — Step 2A: Binary Claim Detection
==========================================
Fine-tunes DistilBERT as a binary classifier on your prepared dataset.

Run:
    python train.py

Output:
    model/claim_detector/          ← saved model + tokenizer (use this in predict.py)
    model/training_log.txt         ← epoch-by-epoch metrics

What this script does:
    1. Loads train.csv and val.csv
    2. Tokenizes with DistilBERT tokenizer
    3. Fine-tunes for 3 epochs with early stopping
    4. Saves the best checkpoint based on validation F1
    5. Prints a final classification report

Expected training time:
    - GPU (even a modest one): ~5–10 minutes
    - CPU only:                ~30–45 minutes
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
TRAIN_PATH   = "data/processed/train.csv"
VAL_PATH     = "data/processed/val.csv"
OUTPUT_DIR   = "model/claim_detector"
LOG_PATH     = "model/training_log.txt"

MAX_LEN      = 128    # Max token length. Debate sentences rarely exceed this.
BATCH_SIZE   = 32     # Reduce to 16 if you run out of memory
EPOCHS       = 3      # 3 is usually enough for fine-tuning on this task
LEARNING_RATE = 2e-5  # Standard for BERT fine-tuning — don't go higher
WARMUP_RATIO  = 0.1   # 10% of steps used for LR warmup

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("model",    exist_ok=True)

# ── Device setup ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {device}")
if device.type == "cuda":
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")


# ═════════════════════════════════════════════════════════════════════════════
# Dataset class
# ═════════════════════════════════════════════════════════════════════════════

class ClaimDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        optimizer.zero_grad()

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        # Gradient clipping — prevents exploding gradients during fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return avg_loss, f1


def eval_epoch(model, loader):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return avg_loss, f1, all_preds, all_labels


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("[Data] Loading train and validation sets...")
    train_df = pd.read_csv(TRAIN_PATH).dropna(subset=["text", "label"])
    val_df   = pd.read_csv(VAL_PATH).dropna(subset=["text", "label"])

    print(f"[Data] Train: {len(train_df)} rows | Val: {len(val_df)} rows")
    print(f"[Data] Train label distribution:\n{train_df.label.value_counts().to_string()}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\n[Model] Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # ── Datasets and loaders ──────────────────────────────────────────────────
    train_dataset = ClaimDataset(train_df, tokenizer, MAX_LEN)
    val_dataset   = ClaimDataset(val_df,   tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"[Model] Loading: {MODEL_NAME} for sequence classification (2 labels)")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    model.to(device)

    # ── Optimizer and scheduler ───────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[Train] Starting training for {EPOCHS} epochs...")
    print(f"[Train] Total steps: {total_steps} | Warmup steps: {warmup_steps}\n")

    best_val_f1   = 0.0
    log_lines     = []

    for epoch in range(1, EPOCHS + 1):
        print(f"── Epoch {epoch}/{EPOCHS} {'─' * 40}")

        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_f1, val_preds, val_labels = eval_epoch(model, val_loader)

        log_line = (
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
        )
        print(log_line)
        log_lines.append(log_line)

        # Save best model based on validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  ✅ New best model saved (Val F1: {val_f1:.4f})")

    # ── Final evaluation on best model ────────────────────────────────────────
    print(f"\n[Eval] Loading best checkpoint for final report...")
    best_model = DistilBertForSequenceClassification.from_pretrained(OUTPUT_DIR)
    best_model.to(device)

    _, _, final_preds, final_labels = eval_epoch(best_model, val_loader)

    report = classification_report(
        final_labels,
        final_preds,
        target_names=["non-claim", "claim"],
        digits=4,
    )

    print("\n[Eval] Final Classification Report:")
    print(report)

    # ── Save training log ──────────────────────────────────────────────────────
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))
        f.write(f"\n\nBest Val F1: {best_val_f1:.4f}\n\n")
        f.write("Final Classification Report:\n")
        f.write(report)

    print(f"\n✅ Training complete.")
    print(f"   Best model saved to: {OUTPUT_DIR}/")
    print(f"   Training log saved to: {LOG_PATH}")
    print(f"\n   Best Validation F1: {best_val_f1:.4f}")
    print(f"\n   Next step: run predict.py to test your classifier.")


if __name__ == "__main__":
    main()