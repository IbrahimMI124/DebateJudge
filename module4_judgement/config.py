import os

# Opt-in lightweight mode for tests / offline environments.
# Default: use full transformer models as specified.
LIGHTWEIGHT_MODE = os.getenv("DEBATEJUDGE_LIGHTWEIGHT", "0") == "1"

# Calibration toggles (defaults preserve the original spec behavior)
# TIME_WEIGHT_MODE:
# - "spec":   0.5 + 0.5 * (position / total)
# - "none":   1.0 (no positional bias)
# - "mild":   1.0 + 0.1 * (position / max(total - 1, 1))
# - "total_minus_1": 0.5 + 0.5 * (position / max(total - 1, 1))
TIME_WEIGHT_MODE = os.getenv("DEBATEJUDGE_TIME_WEIGHT_MODE", "spec").strip().lower()

# If enabled, normalize summed speaker scores by number of statements per speaker.
NORMALIZE_SPEAKER_SCORES = os.getenv("DEBATEJUDGE_NORMALIZE_SCORES", "0") == "1"

# Winner decision calibration (optional)
# If enabled, `decide_winner` can return "Tie" when the top two scores are very close.
RETURN_TIE_ON_CLOSE = os.getenv("DEBATEJUDGE_RETURN_TIE", "0") == "1"
TIE_EPSILON = float(os.getenv("DEBATEJUDGE_TIE_EPSILON", "1e-6"))

RELEVANCE_MODEL_NAME = "all-MiniLM-L6-v2"
NLI_MODEL_NAME = "roberta-large-mnli"


def is_lightweight_mode() -> bool:
	return os.getenv("DEBATEJUDGE_LIGHTWEIGHT", "0") == "1"


def get_time_weight_mode() -> str:
	return os.getenv("DEBATEJUDGE_TIME_WEIGHT_MODE", "spec").strip().lower()


def normalize_speaker_scores_enabled() -> bool:
	return os.getenv("DEBATEJUDGE_NORMALIZE_SCORES", "0") == "1"


def return_tie_enabled() -> bool:
	return os.getenv("DEBATEJUDGE_RETURN_TIE", "0") == "1"


def get_tie_epsilon() -> float:
	try:
		return float(os.getenv("DEBATEJUDGE_TIE_EPSILON", "1e-6"))
	except Exception:
		return 1e-6
