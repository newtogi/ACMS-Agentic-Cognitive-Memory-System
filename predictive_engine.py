"""
Predictive Engine — Agent Brain L6.

Uses pattern detection + MiniCPM5-1B via Ollama to generate predictions from
current state, tracks prediction accuracy over time, and stores everything in
~/agent-brain/predictions.json.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error

from pattern_detector import PatternDetector


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get(
    "AGENT_BRAIN_DIR",
    os.path.expanduser("~/agent-brain"),
))
PREDICTIONS_FILE = DATA_DIR / "predictions.json"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("PREDICTION_MODEL", "openbmb/minicpm5")


# ---------------------------------------------------------------------------
# Predictions store (file-backed JSON)
# ---------------------------------------------------------------------------

def _load_predictions() -> dict:
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"predictions": [], "accuracy_log": []}


def _save_predictions(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Ollama helper
# ---------------------------------------------------------------------------

def _ollama_generate(prompt: str, *, system: str = "", temperature: float = 0.3) -> str:
    """Call Ollama /api/generate and return the response text."""
    payload = json.dumps({
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
            return body.get("response", "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return f"[ollama error: {exc}]"


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class PredictiveEngine:
    """
    L6 Predictive Engine.

    Lifecycle:
        1. Instantiate with episodic memory (list of episode dicts).
        2. Call `engine.predict(current_state)` to get a prediction.
        3. Later call `engine.record_outcome(pred_id, actual_outcome)` to close
           the feedback loop and update accuracy stats.
    """

    def __init__(
        self,
        episodes: list[dict] | None = None,
        *,
        min_pattern_support: int = 2,
        max_ngram: int = 4,
    ):
        self.episodes: list[dict] = episodes or []
        self.detector = PatternDetector(
            self.episodes,
            min_support=min_pattern_support,
            max_ngram=max_ngram,
        )
        # Eagerly scan so confidence denominators are available
        self.detector.scan_sequences()
        self.detector.extract_cause_effect_pairs()

    # ------------------------------------------------------------------ #
    # predict()
    # ------------------------------------------------------------------ #

    def predict(self, current_state: str | dict) -> dict:
        """
        Generate a prediction for the given current state.

        Parameters
        ----------
        current_state : str or dict
            Free-text description of the current situation, or a dict that
            will be serialised to JSON.

        Returns
        -------
        dict with keys:
            prediction_id : str   — UUID for later feedback
            prediction    : str   — the LLM-generated prediction text
            confidence    : float — weighted confidence score (0-1)
            patterns_used : list  — which patterns informed the prediction
            timestamp     : str   — ISO-8601
        """
        if isinstance(current_state, dict):
            state_text = json.dumps(current_state, indent=2)
        else:
            state_text = str(current_state)

        # Gather top patterns & cause-effect pairs
        top_sequences = self.detector.scan_sequences()[:10]
        top_pairs = self.detector.extract_cause_effect_pairs()[:8]

        pattern_summary = self.detector.summarise()
        confidence = self._aggregate_confidence(top_sequences, top_pairs)

        system_prompt = (
            "You are the prediction component of an AI agent brain. "
            "Given the current state and historically observed patterns, "
            "predict the most likely next event or outcome. "
            "Output ONLY a JSON object with exactly these fields:\n"
            '{"reasoning": "<one short sentence>", "prediction": "<the predicted outcome>"}'
        )

        user_prompt = (
            f"CURRENT STATE:\n{state_text}\n\n"
            f"OBSERVED PATTERNS (from episodic memory):\n{pattern_summary}\n\n"
            "Based on the above, what is the most likely next outcome? "
            "Return only the JSON object."
        )

        raw_response = _ollama_generate(
            user_prompt, system=system_prompt, temperature=0.3
        )

        clean_prediction = self._parse_prediction(raw_response)

        pred_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        entry = {
            "prediction_id": pred_id,
            "prediction": clean_prediction,
            "raw_response": raw_response,
            "confidence": round(confidence, 4),
            "patterns_used": [s["id"] for s in top_sequences] + [p["id"] for p in top_pairs],
            "state_snapshot": state_text[:500],
            "timestamp": timestamp,
            "status": "pending",          # pending | confirmed | refuted
            "actual_outcome": None,
            "resolved_at": None,
            "accuracy_delta": None,       # filled on record_outcome
        }

        store = _load_predictions()
        store["predictions"].append(entry)
        _save_predictions(store)

        return {
            "prediction_id": pred_id,
            "prediction": clean_prediction,
            "confidence": entry["confidence"],
            "patterns_used": entry["patterns_used"],
            "timestamp": timestamp,
        }

    @staticmethod
    def _parse_prediction(raw: str) -> str:
        """
        Extract a clean prediction string from the LLM's raw output.

        Tries JSON first ({"reasoning": ..., "prediction": ...}), then
        falls back to looking for a 'PREDICTION:' line, then returns the
        raw text trimmed as a last resort.
        """
        if not raw:
            return ""

        # Try JSON parse
        try:
            data = json.loads(raw.strip())
            if isinstance(data, dict) and "prediction" in data:
                return str(data["prediction"]).strip()
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to find JSON inside code fences or mixed prose
        import re
        json_match = re.search(r"\{[^{}]*\"prediction\"[^{}]*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return str(data["prediction"]).strip()
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: look for PREDICTION: line
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("PREDICTION:"):
                return line[len("PREDICTION:"):].strip()

        # Last resort: return raw trimmed
        return raw.strip()

    # ------------------------------------------------------------------ #
    # record_outcome()
    # ------------------------------------------------------------------ #

    def record_outcome(
        self,
        prediction_id: str,
        actual_outcome: str,
        *,
        evaluation_model: str | None = None,
    ) -> dict:
        """
        Record the actual outcome for a pending prediction and score accuracy.

        Parameters
        ----------
        prediction_id  : str  — the UUID returned by predict()
        actual_outcome : str  — free-text description of what actually happened

        Returns
        -------
        dict with keys:
            prediction_id   : str
            status          : str   — "confirmed" | "refuted" | "partial"
            accuracy_delta  : float — how close the prediction was (0-1)
            feedback_prompt : str   — the prompt used for evaluation
        """
        store = _load_predictions()
        pred = None
        for p in store["predictions"]:
            if p["prediction_id"] == prediction_id:
                pred = p
                break

        if pred is None:
            raise ValueError(f"Unknown prediction_id: {prediction_id}")

        # Ask the LLM to score the prediction vs actual outcome
        eval_prompt = (
            "You are evaluating a prediction against what actually happened.\n\n"
            f"PREDICTION:\n{pred['prediction']}\n\n"
            f"ACTUAL OUTCOME:\n{actual_outcome}\n\n"
            "Rate the prediction accuracy on a scale from 0.0 (completely wrong) "
            "to 1.0 (perfectly correct). Also classify as CONFIRMED (≥0.7), "
            "PARTIAL (0.3–0.7), or REFUTED (<0.3).\n\n"
            "Respond ONLY in this JSON format:\n"
            '{"accuracy": 0.85, "status": "confirmed", "reasoning": "..."}'
        )

        raw = _ollama_generate(eval_prompt, temperature=0.1)

        # Parse evaluation response (best-effort)
        accuracy_delta, status, reasoning = self._parse_evaluation(raw)

        # Update prediction record
        pred["actual_outcome"] = actual_outcome
        pred["status"] = status
        pred["accuracy_delta"] = round(accuracy_delta, 4)
        pred["resolved_at"] = datetime.now(timezone.utc).isoformat()

        # Append to accuracy log
        store["accuracy_log"].append({
            "prediction_id": prediction_id,
            "accuracy": round(accuracy_delta, 4),
            "status": status,
            "reasoning": reasoning,
            "timestamp": pred["resolved_at"],
        })

        _save_predictions(store)

        return {
            "prediction_id": prediction_id,
            "status": status,
            "accuracy_delta": round(accuracy_delta, 4),
            "feedback_prompt": eval_prompt,
        }

    # ------------------------------------------------------------------ #
    # Query helpers
    # ------------------------------------------------------------------ #

    def get_accuracy_stats(self) -> dict:
        """Aggregate accuracy metrics across all resolved predictions."""
        store = _load_predictions()
        resolved = [p for p in store["predictions"] if p["status"] != "pending"]
        if not resolved:
            return {"total": 0, "confirmed": 0, "refuted": 0, "partial": 0,
                    "mean_accuracy": 0.0}

        confirmed = sum(1 for p in resolved if p["status"] == "confirmed")
        refuted   = sum(1 for p in resolved if p["status"] == "refuted")
        partial   = sum(1 for p in resolved if p["status"] == "partial")
        mean_acc  = sum(p["accuracy_delta"] or 0 for p in resolved) / len(resolved)

        return {
            "total": len(resolved),
            "confirmed": confirmed,
            "refuted": refuted,
            "partial": partial,
            "mean_accuracy": round(mean_acc, 4),
        }

    def get_pending_predictions(self) -> list[dict]:
        """Return predictions awaiting outcome recording."""
        store = _load_predictions()
        return [p for p in store["predictions"] if p["status"] == "pending"]

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _aggregate_confidence(
        self,
        sequences: list[dict],
        pairs: list[dict],
    ) -> float:
        """
        Combine individual pattern confidences into a single engine confidence.

        Uses a variant of noisy-OR: P(at least one pattern is relevant) =
        1 − Π(1 − cᵢ), but capped and weighted.
        """
        all_confs = [s["confidence"] for s in sequences] + [p["confidence"] for p in pairs]
        if not all_confs:
            return 0.1  # low default confidence

        # Noisy-OR
        prob_none = 1.0
        for c in all_confs:
            prob_none *= (1.0 - c)

        return max(0.05, min(0.99, 1.0 - prob_none))

    @staticmethod
    def _parse_evaluation(raw: str) -> tuple[float, str, str]:
        """Best-effort parse of LLM evaluation JSON."""
        import re
        json_match = re.search(r'\{[^}]+\}', raw)
        if json_match:
            try:
                obj = json.loads(json_match.group())
                return (
                    float(obj.get("accuracy", 0.5)),
                    str(obj.get("status", "partial")).lower(),
                    str(obj.get("reasoning", "")),
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: look for numeric score
        num_match = re.search(r'(\d+\.?\d*)', raw)
        score = float(num_match.group(1)) if num_match else 0.5
        if score > 1:
            score = score / 100.0  # handle percentage

        if score >= 0.7:
            status = "confirmed"
        elif score >= 0.3:
            status = "partial"
        else:
            status = "refuted"

        return (score, status, raw[:200])


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def load_engine_from_zenbrain(
    zenbrain_episodes: list[dict] | None = None,
    **kwargs: Any,
) -> PredictiveEngine:
    """
    Create a PredictiveEngine from ZenBrain-format episodes.

    Each episode dict should have at least:
        {"content": "...", "timestamp": "...", "layer": "..."}
    """
    if zenbrain_episodes is None:
        zenbrain_episodes = []
    return PredictiveEngine(zenbrain_episodes, **kwargs)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    demo_episodes = [
        {"content": "User asked for Python code and received a working script",
         "timestamp": "2026-09-01T10:00:00Z", "layer": "episodic"},
        {"content": "After receiving code, user requested tests",
         "timestamp": "2026-09-01T10:05:00Z", "layer": "episodic"},
        {"content": "User asked for Python code and got a library module",
         "timestamp": "2026-09-03T14:00:00Z", "layer": "episodic"},
        {"content": "After module delivery, user requested integration tests",
         "timestamp": "2026-09-03T14:10:00Z", "layer": "episodic"},
        {"content": "User asked about API design patterns",
         "timestamp": "2026-09-05T09:00:00Z", "layer": "episodic"},
        {"content": "Because of API pattern discussion, user refactored service layer",
         "timestamp": "2026-09-05T09:30:00Z", "layer": "episodic"},
    ]

    engine = PredictiveEngine(demo_episodes)
    result = engine.predict("User is asking for a new Python module with data processing logic")
    print(json.dumps(result, indent=2))