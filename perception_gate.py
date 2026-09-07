"""
Agent Brain L1 - Perception Gate

Classifies incoming text data into categories and determines salience.
Uses MiniCPM-V 4.6 via Ollama API for classification.
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Literal
from datetime import datetime

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    CATEGORIES,
    SALIENCE_HIGH_THRESHOLD,
    SALIENCE_MEDIUM_THRESHOLD,
    ACTION_THRESHOLDS
)


@dataclass
class PerceptionResult:
    """Result from the perception gate."""
    category: str
    salience: float
    action: str
    reasoning: str
    raw_input: str
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "salience": self.salience,
            "action": self.action,
            "reasoning": self.reasoning,
            "raw_input": self.raw_input,
            "timestamp": self.timestamp
        }


PERCEPTION_PROMPT = """You are a perception filter for an AI agent brain. Analyze the following input data and classify it.

You MUST respond with ONLY a valid JSON object (no markdown, no explanation) with these fields:
- category: one of ["event", "fact", "emotion", "threat", "noise"]
- salience: a number between 0.0 and 1.0 indicating importance
- reasoning: brief explanation of your classification

Category definitions:
- event: Something that happened or is happening (actions, occurrences, state changes)
- fact: Information, knowledge, data points (static information)
- emotion: Emotional content, feelings, sentiment expressed
- threat: Potential danger, risk, warning, urgent issue
- noise: Irrelevant, meaningless, or unprocessable data

Salience guidelines:
- 0.8-1.0: Critical, immediate attention needed
- 0.5-0.8: Important, should be processed
- 0.2-0.5: Moderate relevance
- 0.0-0.2: Low priority, can be deprioritized

Input data to analyze:
{data}"""


def _call_ollama(prompt: str) -> dict:
    """Call Ollama API with the given prompt and return structured response."""
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 256
        }
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            response_text = result.get('response', '{}')
            return json.loads(response_text)
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect to Ollama at {OLLAMA_BASE_URL}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from model: {e}")


def _determine_action(salience: float) -> str:
    """Determine action based on salience score."""
    if salience >= SALIENCE_HIGH_THRESHOLD:
        return "escalate"
    elif salience >= SALIENCE_MEDIUM_THRESHOLD:
        return "encode"
    else:
        return "discard"


def perceive(data: str, *, profile: str = "system-bot", persist: bool = True) -> dict:
    """
    Analyze input data through the perception gate.

    Args:
        data: Input text to analyze
        profile: ZenBrain profile to write to (default: system-bot)
        persist: If True and salience >= medium threshold, store as an
                 episodic memory in ZenBrain so all bots can recall it later.

    Returns:
        dict with keys: category, salience, action, reasoning, raw_input,
                        timestamp, and (if persisted) memory_id.
    """
    prompt = PERCEPTION_PROMPT.format(data=data)

    try:
        model_output = _call_ollama(prompt)
    except (ConnectionError, ValueError) as e:
        # Fallback to noise classification if model unavailable
        return PerceptionResult(
            category="noise",
            salience=0.1,
            action="discard",
            reasoning=f"Model unavailable: {str(e)}",
            raw_input=data,
            timestamp=datetime.now().isoformat()
        ).to_dict()

    # Extract and validate fields
    category = model_output.get("category", "noise")
    if category not in CATEGORIES:
        category = "noise"

    salience = float(model_output.get("salience", 0.5))
    salience = max(0.0, min(1.0, salience))  # Clamp to [0, 1]

    action = _determine_action(salience)
    reasoning = model_output.get("reasoning", "No reasoning provided")

    result = PerceptionResult(
        category=category,
        salience=salience,
        action=action,
        reasoning=reasoning,
        raw_input=data,
        timestamp=datetime.now().isoformat()
    )

    out = result.to_dict()

    # Persist to ZenBrain if salience is high enough
    if persist and salience >= SALIENCE_MEDIUM_THRESHOLD:
        try:
            from zenbrain_client import store_episode
            memory_id = store_episode(
                content=f"[{category}] {data}",
                profile=profile,
                emotional_weight=salience,
                metadata={
                    "source": "agent_brain_perception_gate",
                    "category": category,
                    "reasoning": reasoning,
                },
            )
            out["memory_id"] = memory_id
        except Exception as exc:
            out["persist_error"] = str(exc)

    return out


if __name__ == "__main__":
    # Example usage
    test_inputs = [
        "The server just crashed and all users are offline",
        "The user prefers dark mode for the interface",
        "I feel anxious about the upcoming deadline",
        "Someone is trying to brute force the SSH port",
        "asdkjfh askdjfh 12345",
    ]

    for text in test_inputs:
        result = perceive(text)
        print(f"\nInput: {text[:50]}...")
        print(f"Category: {result['category']}")
        print(f"Salience: {result['salience']}")
        print(f"Action: {result['action']}")
        print(f"Reasoning: {result['reasoning']}")
