from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .argument_relation_prompt import build_argument_relation_prompt


ALLOWED_LABELS = {
    "supports",
    "evidence_for",
    "rebuts",
    "contradicts",
    "undercuts",
    "counterexample",
    "qualifies",
    "concedes",
    "clarifies",
}

# Test-phase constant scores per label.
# (You said you'll move to configured weights later.)
LABEL_SCORES: dict[str, float] = {
    "supports": 1.0,
    "evidence_for": 1.0,
    "rebuts": -1.0,
    "contradicts": -1.0,
    "undercuts": -0.7,
    "counterexample": -0.8,
    "qualifies": 0.2,
    "concedes": 0.0,
    "clarifies": 0.0,
}


def score_for_label(label: str) -> float:
    return float(LABEL_SCORES.get((label or "").strip().lower(), 0.0))


def map_rich_label_to_nli(label: str) -> str:
    """Map the richer argument-relation label space into 3-way NLI labels.

    This keeps the rest of the pipeline (which expects CONTRADICTION/NEUTRAL/ENTAILMENT)
    unchanged.
    """

    lab = (label or "").strip().lower()
    if lab in {"supports", "evidence_for"}:
        return "ENTAILMENT"
    if lab in {"rebuts", "contradicts", "undercuts", "counterexample"}:
        return "CONTRADICTION"
    if lab in {"qualifies", "concedes", "clarifies"}:
        return "NEUTRAL"
    return "NEUTRAL"


def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None

    # First try: find a JSON object with a simple regex (non-greedy).
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        return m.group(0)

    # Fallback: bracket scan.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return None


def _normalize_result(obj: Any) -> Dict[str, Any]:
    # Safety fallback
    if not isinstance(obj, dict):
        return {"label": "clarifies", "confidence": 0.0, "rationale": "parse_error"}

    label = str(obj.get("label", "")).strip().lower()
    if label not in ALLOWED_LABELS:
        label = "clarifies"

    conf = obj.get("confidence", 0.0)
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.0
    conf_f = max(0.0, min(1.0, conf_f))

    rationale = str(obj.get("rationale", "")).strip()
    # Ensure <= 30 words.
    words = rationale.split()
    if len(words) > 30:
        rationale = " ".join(words[:30])

    return {"label": label, "confidence": conf_f, "rationale": rationale}


@dataclass
class QwenLocalConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    cache_dir: Optional[str] = None
    max_new_tokens: int = 160
    device_map: Optional[str] = None


_MODEL = None
_TOKENIZER = None
_LAST_LOAD_ERROR: Optional[str] = None


def _get_cfg() -> QwenLocalConfig:
    model_name = (os.getenv("DEBATEJUDGE_QWEN_MODEL") or "Qwen/Qwen2.5-7B-Instruct").strip()
    cache_dir = (os.getenv("DEBATEJUDGE_HF_CACHE_DIR") or "").strip() or None
    max_new_tokens = int(os.getenv("DEBATEJUDGE_QWEN_MAX_NEW_TOKENS") or "160")
    device_map = (os.getenv("DEBATEJUDGE_QWEN_DEVICE_MAP") or "").strip()
    device_map_opt = device_map if device_map else None
    return QwenLocalConfig(
        model_name=model_name,
        cache_dir=cache_dir,
        max_new_tokens=max_new_tokens,
        device_map=device_map_opt,
    )


def _lazy_load_model() -> tuple[Any, Any, QwenLocalConfig]:
    global _MODEL, _TOKENIZER

    global _LAST_LOAD_ERROR
    _LAST_LOAD_ERROR = None

    cfg = _get_cfg()
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER, cfg

    # If the user provided a filesystem path and it doesn't exist, fail fast with a clear message.
    # Otherwise, transformers will treat it as a HF repo id and produce a confusing error.
    model_name = (cfg.model_name or "").strip()
    if model_name:
        expanded = Path(model_name).expanduser()
        looks_like_path = model_name.startswith(("/", "./", "../", "~"))
        if looks_like_path and not expanded.exists():
            _LAST_LOAD_ERROR = f"local_path_not_found:{expanded}"
            return None, None, cfg

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception:
        _LAST_LOAD_ERROR = "transformers_not_installed"
        return None, None, cfg

    accelerate_available = True
    try:
        import accelerate  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        accelerate_available = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir, trust_remote_code=True)

        model_kwargs: dict[str, Any] = {
            "cache_dir": cfg.cache_dir,
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }

        # `device_map="auto"` requires accelerate. If it's not installed, fall back
        # to a normal load (usually CPU) instead of failing.
        if cfg.device_map and accelerate_available:
            model_kwargs["device_map"] = cfg.device_map

        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
    except Exception as e:
        msg = str(e).strip().replace("\n", " ")
        if len(msg) > 160:
            msg = msg[:160] + "..."
        _LAST_LOAD_ERROR = f"load_error:{e.__class__.__name__}:{msg}" if msg else f"load_error:{e.__class__.__name__}"
        return None, None, cfg

    _TOKENIZER = tokenizer
    _MODEL = model
    return model, tokenizer, cfg


def classify_argument_relation(statement_a: str, statement_b: str) -> Dict[str, Any]:
    """Classify the relation of B to A using a local Qwen2.5 model.

    Returns strict JSON dict:
      {"label": <allowed>, "confidence": float, "rationale": str}

    If the model/deps are unavailable, returns a safe fallback.
    """

    model, tokenizer, cfg = _lazy_load_model()
    if model is None or tokenizer is None:
        reason = _LAST_LOAD_ERROR or "model_unavailable"
        if cfg.device_map and (reason == "load_error" or reason.startswith("load_error:")):
            # A very common failure mode: device_map auto requested but accelerate missing.
            try:
                import importlib.util

                if importlib.util.find_spec("accelerate") is None:
                    reason = "load_error_missing_accelerate"
            except Exception:
                pass
        return {"label": "clarifies", "confidence": 0.0, "rationale": reason}

    prompt = build_argument_relation_prompt(statement_a, statement_b)

    # Prefer chat template if available.
    messages = [
        {"role": "system", "content": "You are an expert argumentation analyst."},
        {"role": "user", "content": prompt},
    ]

    try:
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    try:
        # Move inputs to model device (works for both CPU and device_map).
        input_ids = input_ids.to(model.device)
    except Exception:
        pass

    try:
        gen = model.generate(
            input_ids,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    except Exception:
        return {"label": "clarifies", "confidence": 0.0, "rationale": "generation_error"}

    # Decode only the newly generated portion when possible.
    try:
        out_tokens = gen[0][input_ids.shape[-1] :]
        text = tokenizer.decode(out_tokens, skip_special_tokens=True)
    except Exception:
        text = tokenizer.decode(gen[0], skip_special_tokens=True)

    json_str = _extract_first_json_object(text)
    if not json_str:
        return {"label": "clarifies", "confidence": 0.0, "rationale": "no_json"}

    try:
        obj = json.loads(json_str)
    except Exception:
        return {"label": "clarifies", "confidence": 0.0, "rationale": "json_parse_error"}

    return _normalize_result(obj)
