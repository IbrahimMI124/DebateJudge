import os
import sys
import unittest
from pathlib import Path


# Ensure unit tests don't require downloading large models
os.environ.setdefault("DEBATEJUDGE_LIGHTWEIGHT", "1")

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class TestPipeline(unittest.TestCase):
    def test_run_judgement_pipeline_keys_and_values(self):
        from module4_judgement.main import run_judgement

        statements = [
            {"id": 2, "speaker": "B", "text": "We should ban cars in cities."},
            {"id": 1, "speaker": "A", "text": "Cars cause pollution and traffic."},
            {"id": 3, "speaker": "A", "text": "Actually, cars do not cause any pollution."},
        ]

        claims = [
            {"id": 1, "speaker": "A", "has_evidence": True, "claim_type": "statistical", "confidence": 0.9},
            # Missing optional fields should be handled
            {"id": 2, "speaker": "B"},
        ]

        facts = [
            {"id": 1, "factual_score": 1.0, "confidence": 0.95},
            {"id": 2, "factual_score": 0.6},
            # Fact for missing statement id should be ignored safely
            {"id": 999, "factual_score": 0.1},
        ]

        out = run_judgement(statements, claims, facts, topic="Urban transport policy")

        self.assertIsInstance(out, dict)
        for key in ["speaker_scores", "consistency", "winner", "explanation"]:
            self.assertIn(key, out)

        self.assertIn(out["winner"], out["speaker_scores"])

        # Scores should exist and be positive
        for score in out["speaker_scores"].values():
            self.assertGreater(score, 0)

        # Consistency scores should be within [0, 1]
        for c in out["consistency"].values():
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)

        self.assertIsInstance(out["explanation"], str)
        self.assertIn("Speaker A Score", out["explanation"])
        self.assertIn("Speaker B Score", out["explanation"])


if __name__ == "__main__":
    unittest.main()
