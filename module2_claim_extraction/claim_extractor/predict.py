import json, os, torch, torch.nn as nn, torch.nn.functional as F
from typing import Any, Dict
from transformers import AutoTokenizer, AutoModel


DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE      = torch.device(DEVICE_TYPE)

CKPT_DIR = "claim_extractor"
MODEL_FILE = os.path.join(CKPT_DIR, "model_1.pt")

GDRIVE_URL = "https://drive.google.com/uc?id=1vZF8v_0Exp2xZmJxGCL3dQ25MDZ0R1ny"



def _download_if_needed():
    if not os.path.exists(MODEL_FILE):
        print("Model not found locally. Downloading from Google Drive...")

        os.makedirs(CKPT_DIR, exist_ok=True)

        try:
            import gdown
        except ImportError:
            raise RuntimeError("Please install gdown: pip install gdown")

        gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)

        print("Download complete.")
    else:
        print("Model found locally. Skipping download.")


_download_if_needed()



with open(os.path.join(CKPT_DIR, "config.json")) as f:
    _cfg = json.load(f)

_MODEL_NAME  = _cfg["model_name"]
_MAX_LENGTH  = _cfg["max_length"]
_MLP_HIDDEN  = _cfg["mlp_hidden"]
_DROPOUT     = _cfg["dropout"]
_NER_TAGS    = _cfg["ner_tags"]
_ATTRIBUTES  = _cfg["attributes"]
_RELATIONS   = _cfg["relations"]
_CLAIM_TYPES = _cfg["claim_types"]
_STANCES     = _cfg["stances"]

_ID2NER   = {i: t for i, t in enumerate(_NER_TAGS)}
_ID2ATTR  = {i: a for i, a in enumerate(_ATTRIBUTES)}
_ID2REL   = {i: r for i, r in enumerate(_RELATIONS)}
_ID2CT    = {i: c for i, c in enumerate(_CLAIM_TYPES)}
_ID2STANCE = {i: s for i, s in enumerate(_STANCES)}

_tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True)

class _ContextualAttentionPooling(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, seq, mask):
        s = self.proj(seq).squeeze(-1).masked_fill(mask == 0, float("-inf"))
        return (torch.softmax(s, -1).unsqueeze(-1) * seq).sum(1)


class _ClassificationHead(nn.Module):
    def __init__(self, in_d, mid_d, out_d, drop):
        super().__init__()
        self.dense = nn.Linear(in_d, mid_d)
        self.norm  = nn.LayerNorm(mid_d)
        self.act   = nn.GELU()
        self.drops = nn.ModuleList([nn.Dropout(drop) for _ in range(5)])
        self.out   = nn.Linear(mid_d, out_d)

    def forward(self, x):
        x = self.act(self.norm(self.dense(x)))
        return sum(self.out(d(x)) for d in self.drops) / len(self.drops)


class _MultiTaskClaimModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(_MODEL_NAME)
        h = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(_DROPOUT)
        self.attention_pool = _ContextualAttentionPooling(h)

        self.ner_linear = nn.Linear(h, len(_NER_TAGS))
        self.attr_head   = _ClassificationHead(h, _MLP_HIDDEN, len(_ATTRIBUTES), _DROPOUT)
        self.rel_head    = _ClassificationHead(h, _MLP_HIDDEN, len(_RELATIONS), _DROPOUT)
        self.ct_head     = _ClassificationHead(h, _MLP_HIDDEN, len(_CLAIM_TYPES), _DROPOUT)
        self.stance_head = _ClassificationHead(h, _MLP_HIDDEN, len(_STANCES), _DROPOUT)

    def forward(self, input_ids, attention_mask):
        seq = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.dropout(self.attention_pool(seq, attention_mask))

        return {
            "ner_emissions": self.ner_linear(self.dropout(seq)),
            "attr": self.attr_head(pooled),
            "rel": self.rel_head(pooled),
            "ct": self.ct_head(pooled),
            "stance": self.stance_head(pooled),
        }


_infer_model = _MultiTaskClaimModel().to(DEVICE)

_infer_model.load_state_dict(
    torch.load(MODEL_FILE, map_location=DEVICE)
)

_infer_model.eval()
print(f"Loaded model from {MODEL_FILE} on {DEVICE}")


@torch.no_grad()
def infer(text: str, speaker: str = "?", uid: int = 0) -> Dict[str, Any]:
    enc = _tokenizer(
        text,
        max_length=_MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    with torch.amp.autocast(DEVICE_TYPE):
        logits = _infer_model(ids, mask)

    return {
        "id": uid,
        "speaker": speaker,
        "text": text,
        "attribute": _ID2ATTR[logits["attr"][0].argmax(-1).item()],
        "relation": _ID2REL[logits["rel"][0].argmax(-1).item()],
        "claim_type": _ID2CT[logits["ct"][0].argmax(-1).item()],
        "stance": _ID2STANCE[logits["stance"][0].argmax(-1).item()],
    }