from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
	value = os.getenv(name)
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return float(value)
	except Exception:
		return default


def _env_float_optional(name: str) -> float | None:
	value = os.getenv(name)
	if value is None:
		return None
	try:
		return float(value)
	except Exception:
		return None


def _env_str(name: str, default: str) -> str:
	value = os.getenv(name)
	return default if value is None else value


def _env_str_optional(name: str) -> str | None:
	value = os.getenv(name)
	if value is None:
		return None
	return value


def _env_bool_optional(name: str) -> bool | None:
	value = os.getenv(name)
	if value is None:
		return None
	return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# ------------------------------
# Optional JSON config
# ------------------------------
# Set `DEBATEJUDGE_CONFIG_JSON` to a JSON file path to configure weights/settings
# without environment variables. Precedence:
#   env var > JSON config > defaults

_JSON_CACHE: dict[str, Any] = {
	"path": None,
	"mtime": None,
	"data": None,
}


def _get_config_json_path() -> str | None:
	path = os.getenv("DEBATEJUDGE_CONFIG_JSON") or os.getenv("DEBATEJUDGE_CONFIG_PATH")
	if not path:
		return None
	return path.strip()


def _load_json_config() -> dict[str, Any]:
	path_str = _get_config_json_path()
	if not path_str:
		return {}

	try:
		path = Path(path_str)
		if not path.is_file():
			return {}
		mtime = path.stat().st_mtime
		if _JSON_CACHE["path"] == str(path) and _JSON_CACHE["mtime"] == mtime and _JSON_CACHE["data"] is not None:
			return _JSON_CACHE["data"]

		data = json.loads(path.read_text(encoding="utf-8"))
		if not isinstance(data, dict):
			data = {}
		_JSON_CACHE.update({"path": str(path), "mtime": mtime, "data": data})
		return data
	except Exception:
		return {}


def _json_get(path: list[str], default: Any = None) -> Any:
	data: Any = _load_json_config()
	for key in path:
		if not isinstance(data, dict) or key not in data:
			return default
		data = data[key]
	return data


def _json_float(path: list[str]) -> float | None:
	value = _json_get(path, None)
	if value is None:
		return None
	try:
		return float(value)
	except Exception:
		return None


def _json_bool(path: list[str]) -> bool | None:
	value = _json_get(path, None)
	if value is None:
		return None
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return bool(value)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "y", "on"}
	return None


def _json_str(path: list[str]) -> str | None:
	value = _json_get(path, None)
	if value is None:
		return None
	return str(value)


# Opt-in lightweight mode for tests / offline environments.
# Default: use full transformer models as specified.
LIGHTWEIGHT_MODE = _env_bool("DEBATEJUDGE_LIGHTWEIGHT", False)

# Model identifiers (per spec)
RELEVANCE_MODEL_NAME = "all-MiniLM-L6-v2"
NLI_MODEL_NAME = "roberta-large-mnli"


# ------------------------------
# Scoring hyperparameters
# ------------------------------
# These defaults preserve the original spec behavior implemented in `scoring.py`.
# Override via env vars to reweight contributions.

SCORING_FACT_DEFAULT = 0.5
SCORING_FACT_WEIGHT = 0.4
SCORING_RELEVANCE_WEIGHT = 0.2
SCORING_EVIDENCE_BONUS = 0.2
SCORING_BASE_BONUS = 0.2


def get_scoring_fact_default() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_SCORING_FACT_DEFAULT")
	if env_value is not None:
		return env_value
	json_value = _json_float(["scoring", "fact_default"])
	if json_value is not None:
		return json_value
	return SCORING_FACT_DEFAULT


def get_scoring_fact_weight() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_SCORING_FACT_WEIGHT")
	if env_value is not None:
		return env_value
	json_value = _json_float(["scoring", "fact_weight"])
	if json_value is not None:
		return json_value
	return SCORING_FACT_WEIGHT


def get_scoring_relevance_weight() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_SCORING_RELEVANCE_WEIGHT")
	if env_value is not None:
		return env_value
	json_value = _json_float(["scoring", "relevance_weight"])
	if json_value is not None:
		return json_value
	return SCORING_RELEVANCE_WEIGHT


def get_scoring_evidence_bonus() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_SCORING_EVIDENCE_BONUS")
	if env_value is not None:
		return env_value
	json_value = _json_float(["scoring", "evidence_bonus"])
	if json_value is not None:
		return json_value
	return SCORING_EVIDENCE_BONUS


def get_scoring_base_bonus() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_SCORING_BASE_BONUS")
	if env_value is not None:
		return env_value
	json_value = _json_float(["scoring", "base_bonus"])
	if json_value is not None:
		return json_value
	return SCORING_BASE_BONUS


# ------------------------------
# Time-weight calibration
# ------------------------------
# TIME_WEIGHT_MODE:
# - "spec":         spec_base + spec_scale * (position / max(total, min_denom))
# - "none":         constant (no positional bias)
# - "mild":         mild_base + mild_scale * (position / max(total - 1, min_denom))
# - "total_minus_1":tm1_base  + tm1_scale  * (position / max(total - 1, min_denom))

TIME_WEIGHT_MODE = _env_str("DEBATEJUDGE_TIME_WEIGHT_MODE", "spec").strip().lower()

TIME_WEIGHT_NONE_VALUE = 1.0
TIME_WEIGHT_MIN_DENOM = 1.0

TIME_WEIGHT_SPEC_BASE = 0.5
TIME_WEIGHT_SPEC_SCALE = 0.5

TIME_WEIGHT_MILD_BASE = 1.0
TIME_WEIGHT_MILD_SCALE = 0.1

TIME_WEIGHT_TOTAL_MINUS_1_BASE = 0.5
TIME_WEIGHT_TOTAL_MINUS_1_SCALE = 0.5


def is_lightweight_mode() -> bool:
	return _env_bool("DEBATEJUDGE_LIGHTWEIGHT", False)


def get_time_weight_mode() -> str:
	env_value = _env_str_optional("DEBATEJUDGE_TIME_WEIGHT_MODE")
	if env_value is not None:
		return env_value.strip().lower()
	json_value = _json_str(["time_weight", "mode"])
	if json_value is not None:
		return json_value.strip().lower()
	return "spec"


def get_time_weight_none_value() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_TIME_WEIGHT_NONE_VALUE")
	if env_value is not None:
		return env_value
	json_value = _json_float(["time_weight", "none_value"])
	if json_value is not None:
		return json_value
	return TIME_WEIGHT_NONE_VALUE


def get_time_weight_min_denom() -> float:
	# Used for both `total` and `total - 1` denominators.
	env_value = _env_float_optional("DEBATEJUDGE_TIME_WEIGHT_MIN_DENOM")
	if env_value is not None:
		return env_value
	json_value = _json_float(["time_weight", "min_denom"])
	if json_value is not None:
		return json_value
	return TIME_WEIGHT_MIN_DENOM


def get_time_weight_spec_base() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_TIME_WEIGHT_SPEC_BASE")
	if env_value is not None:
		return env_value
	json_value = _json_float(["time_weight", "spec", "base"])
	if json_value is not None:
		return json_value
	return TIME_WEIGHT_SPEC_BASE


def get_time_weight_spec_scale() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_TIME_WEIGHT_SPEC_SCALE")
	if env_value is not None:
		return env_value
	json_value = _json_float(["time_weight", "spec", "scale"])
	if json_value is not None:
		return json_value
	return TIME_WEIGHT_SPEC_SCALE


def get_time_weight_mild_base() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_TIME_WEIGHT_MILD_BASE")
	if env_value is not None:
		return env_value
	json_value = _json_float(["time_weight", "mild", "base"])
	if json_value is not None:
		return json_value
	return TIME_WEIGHT_MILD_BASE


def get_time_weight_mild_scale() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_TIME_WEIGHT_MILD_SCALE")
	if env_value is not None:
		return env_value
	json_value = _json_float(["time_weight", "mild", "scale"])
	if json_value is not None:
		return json_value
	return TIME_WEIGHT_MILD_SCALE


def get_time_weight_total_minus_1_base() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_TIME_WEIGHT_TOTAL_MINUS_1_BASE")
	if env_value is not None:
		return env_value
	json_value = _json_float(["time_weight", "total_minus_1", "base"])
	if json_value is not None:
		return json_value
	return TIME_WEIGHT_TOTAL_MINUS_1_BASE


def get_time_weight_total_minus_1_scale() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_TIME_WEIGHT_TOTAL_MINUS_1_SCALE")
	if env_value is not None:
		return env_value
	json_value = _json_float(["time_weight", "total_minus_1", "scale"])
	if json_value is not None:
		return json_value
	return TIME_WEIGHT_TOTAL_MINUS_1_SCALE


# ------------------------------
# Other calibration toggles
# ------------------------------

# If enabled, normalize summed speaker scores by number of statements per speaker.
NORMALIZE_SPEAKER_SCORES = _env_bool("DEBATEJUDGE_NORMALIZE_SCORES", False)


def normalize_speaker_scores_enabled() -> bool:
	env_value = _env_bool_optional("DEBATEJUDGE_NORMALIZE_SCORES")
	if env_value is not None:
		return env_value
	json_value = _json_bool(["normalization", "normalize_speaker_scores"])
	if json_value is not None:
		return json_value
	return False


# Winner decision calibration (optional)
# If enabled, `decide_winner` can return "Tie" when the top two scores are very close.
RETURN_TIE_ON_CLOSE = _env_bool("DEBATEJUDGE_RETURN_TIE", False)
TIE_EPSILON = float(_env_str("DEBATEJUDGE_TIE_EPSILON", "1e-6"))


def return_tie_enabled() -> bool:
	env_value = _env_bool_optional("DEBATEJUDGE_RETURN_TIE")
	if env_value is not None:
		return env_value
	json_value = _json_bool(["winner", "return_tie"])
	if json_value is not None:
		return json_value
	return False


def get_tie_epsilon() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_TIE_EPSILON")
	if env_value is not None:
		return env_value
	json_value = _json_float(["winner", "tie_epsilon"])
	if json_value is not None:
		return json_value
	return 1e-6


# ------------------------------
# Rebuttal awareness (cross-speaker)
# ------------------------------

REBUTTAL_ENABLED_DEFAULT = False
REBUTTAL_SIMILARITY_THRESHOLD_DEFAULT = 0.6
REBUTTAL_BONUS_DEFAULT = 0.1
REBUTTAL_CONTRADICTION_BONUS_DEFAULT = REBUTTAL_BONUS_DEFAULT
REBUTTAL_AGREEMENT_BONUS_DEFAULT = 0.0
REBUTTAL_SOFT_THRESHOLD_BAND_DEFAULT = 0.0
REBUTTAL_SOFT_THRESHOLD_WEIGHT_DEFAULT = 0.5


def rebuttal_enabled() -> bool:
	env_value = _env_bool_optional("DEBATEJUDGE_REBUTTAL_ENABLED")
	if env_value is not None:
		return env_value
	json_value = _json_bool(["rebuttal", "enabled"])
	if json_value is not None:
		return json_value
	return REBUTTAL_ENABLED_DEFAULT


def get_rebuttal_similarity_threshold() -> float:
	env_value = _env_float_optional("DEBATEJUDGE_REBUTTAL_SIMILARITY_THRESHOLD")
	if env_value is not None:
		return env_value
	json_value = _json_float(["rebuttal", "similarity_threshold"])
	if json_value is not None:
		return json_value
	return REBUTTAL_SIMILARITY_THRESHOLD_DEFAULT


def get_rebuttal_soft_threshold_band() -> float:
	"""Optional soft band below the similarity threshold.

	If > 0:
	- sim >= threshold                -> weight 1.0
	- threshold - band <= sim < thr   -> weight `soft_threshold_weight`
	- sim < threshold - band          -> weight 0.0

	Default is 0.0, which preserves the original hard threshold.
	"""

	env_value = _env_float_optional("DEBATEJUDGE_REBUTTAL_SOFT_THRESHOLD_BAND")
	if env_value is not None:
		return env_value
	json_value = _json_float(["rebuttal", "soft_threshold_band"])
	if json_value is not None:
		return json_value
	return REBUTTAL_SOFT_THRESHOLD_BAND_DEFAULT


def get_rebuttal_soft_threshold_weight() -> float:
	"""Weight applied inside the soft band (see `get_rebuttal_soft_threshold_band`)."""

	env_value = _env_float_optional("DEBATEJUDGE_REBUTTAL_SOFT_THRESHOLD_WEIGHT")
	if env_value is not None:
		return env_value
	json_value = _json_float(["rebuttal", "soft_threshold_weight"])
	if json_value is not None:
		return json_value
	return REBUTTAL_SOFT_THRESHOLD_WEIGHT_DEFAULT


def get_rebuttal_bonus() -> float:
	"""Backward-compatible alias.

	Historically rebuttal used a single flat bonus (`rebuttal.bonus`). The newer
	NLI-based logic uses separate `contradiction_bonus` and `agreement_bonus`.
	This function returns the contradiction bonus.
	"""
	return get_rebuttal_contradiction_bonus()


def get_rebuttal_contradiction_bonus() -> float:
	"""Base coefficient for contradiction rebuttals.

	Final applied bonus scales by similarity: bonus = coeff * similarity.
	"""
	env_value = _env_float_optional("DEBATEJUDGE_REBUTTAL_CONTRADICTION_BONUS")
	if env_value is not None:
		return env_value
	json_value = _json_float(["rebuttal", "contradiction_bonus"])
	if json_value is not None:
		return json_value

	# Backward-compat: fall back to the older single bonus.
	legacy_env = _env_float_optional("DEBATEJUDGE_REBUTTAL_BONUS")
	if legacy_env is not None:
		return legacy_env
	legacy_json = _json_float(["rebuttal", "bonus"])
	if legacy_json is not None:
		return legacy_json

	return REBUTTAL_CONTRADICTION_BONUS_DEFAULT


def get_rebuttal_agreement_bonus() -> float:
	"""Base coefficient for agreement/entailment interactions.

	Final applied bonus scales by similarity: bonus = coeff * similarity.
	"""
	env_value = _env_float_optional("DEBATEJUDGE_REBUTTAL_AGREEMENT_BONUS")
	if env_value is not None:
		return env_value
	json_value = _json_float(["rebuttal", "agreement_bonus"])
	if json_value is not None:
		return json_value
	return REBUTTAL_AGREEMENT_BONUS_DEFAULT
