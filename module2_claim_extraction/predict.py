import torch
import torch.nn as nn
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import json

MODEL_ID = "SyedNaveed/claim-extraction"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = hf_hub_download(MODEL_ID, "config.json")
with open(config_path, "r") as f:
    CONFIG = json.load(f)


class ContextualAttentionPooling(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, seq, mask):
        scores = self.proj(seq).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        return (torch.softmax(scores, dim=-1).unsqueeze(-1) * seq).sum(1)


class ClassificationHead(nn.Module):
    def __init__(self, in_d, mid_d, out_d, drop):
        super().__init__()
        self.dense = nn.Linear(in_d, mid_d)
        self.norm = nn.LayerNorm(mid_d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(mid_d, out_d)

    def forward(self, x):
        x = self.act(self.norm(self.dense(x)))
        x = self.drop(x)
        return self.out(x)


class MultiTaskClaimModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            config["model_name"],
            torch_dtype=torch.float32
        )

        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(config["dropout"])
        self.attention_pool = ContextualAttentionPooling(hidden)

        self.ner_linear = nn.Linear(hidden, len(config["ner_tags"]))

        self.attr_head = ClassificationHead(hidden, config["mlp_hidden"], len(config["attributes"]), config["dropout"])
        self.rel_head  = ClassificationHead(hidden, config["mlp_hidden"], len(config["relations"]), config["dropout"])
        self.ct_head   = ClassificationHead(hidden, config["mlp_hidden"], len(config["claim_types"]), config["dropout"])
        self.stance_head = ClassificationHead(hidden, config["mlp_hidden"], len(config["stances"]), config["dropout"])

    def forward(self, input_ids, attention_mask):
        seq = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.dropout(self.attention_pool(seq, attention_mask))

        return {
            "ner": self.ner_linear(self.dropout(seq)),
            "attr": self.attr_head(pooled),
            "rel": self.rel_head(pooled),
            "ct": self.ct_head(pooled),
            "stance": self.stance_head(pooled),
        }


_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    fix_mistral_regex=True
)

_model = MultiTaskClaimModel(CONFIG).to(DEVICE).float()

model_path = hf_hub_download(MODEL_ID, "model_1.pt")

_model.load_state_dict(
    torch.load(model_path, map_location=DEVICE, weights_only=True)
)

_model.eval()

print(f"Model loaded from HF on {DEVICE}")

def _decode_entities(tokens, pred_ids, mask, word_ids):
    entities = []
    current_tokens = []
    prev_word_id = None

    for token, idx, m, word_id in zip(tokens, pred_ids, mask, word_ids):

        if m == 0:
            break

        if word_id is None:
            continue

        tag = CONFIG["ner_tags"][idx]

        if word_id != prev_word_id:
            if current_tokens:
                entities.append(_tokenizer.convert_tokens_to_string(current_tokens).strip())
                current_tokens = []

        if tag.startswith("B-"):
            if current_tokens:
                entities.append(_tokenizer.convert_tokens_to_string(current_tokens).strip())
            current_tokens = [token]

        elif tag.startswith("I-") and current_tokens:
            current_tokens.append(token)

        else:
            if current_tokens:
                entities.append(_tokenizer.convert_tokens_to_string(current_tokens).strip())
                current_tokens = []

        prev_word_id = word_id

    if current_tokens:
        entities.append(_tokenizer.convert_tokens_to_string(current_tokens).strip())

    return list(dict.fromkeys(e for e in entities if e))  


@torch.no_grad()
def predict(text: str) -> Dict[str, Any]:

    enc = _tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=CONFIG["max_length"],
        return_tensors="pt",
        return_offsets_mapping=True
    )

    word_ids = enc.word_ids(batch_index=0)

    input_ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    logits = _model(input_ids, mask)

    
    ner_ids = logits["ner"][0].argmax(-1).cpu().tolist()
    tokens = _tokenizer.convert_ids_to_tokens(input_ids[0])
    mask_list = mask[0].cpu().tolist()

    entities = _decode_entities(tokens, ner_ids, mask_list, word_ids)

    
    attr = CONFIG["attributes"][logits["attr"][0].argmax().item()]
    rel = CONFIG["relations"][logits["rel"][0].argmax().item()]
    ct = CONFIG["claim_types"][logits["ct"][0].argmax().item()]
    stance = CONFIG["stances"][logits["stance"][0].argmax().item()]

    return {
        "text": text,
        "entities": entities,
        "attribute": attr,
        "relation": rel,
        "claim_type": ct,
        "stance": stance
    }

if __name__ == "__main__":
    test = "Neuer has more saves than Buffon"
    print(predict(test))