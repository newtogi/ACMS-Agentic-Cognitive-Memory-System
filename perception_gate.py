"""
Agent Brain L1 - Perception Gate

Classifies incoming text data into categories and determines salience.
Uses qwen2.5:7b via Ollama API for classification.
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


# Evolved prompt from darwinian-evolver (100% on 8 edge cases)
PERCEPTION_PROMPT = """Look at this text and tell me what it is.

{data}

Return a JSON answer with the following structure:
{{
  "category": "{{category}}",
  "salience": {{salience}},
  "action": "{{action}}",
  "reasoning": "{{reasoning}}"
}}
Remember to use only these categories: episodic, semantic, procedural, emotional, perceptual, social, noise.
Use only these actions: persist, discard, escalate, defer.
Keep salience between 0.0 and 1.0."""


def _call_ollama(prompt: str) -> dict:
    """Call Ollama API with the given prompt and return structured response."""
    url = f"{OLLAMA_BASE_URL}/v1/chat/completions"

    payload = {
        "model": OLLAMA_MODEL,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": "You are a JSON classifier. Return ONLY a valid JSON object. No markdown, no explanation."},
            {"role": "user", "content": prompt},
        ],
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            raw = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Extract JSON from response (handle markdown fences)
            clean = raw.replace("```json", "").replace("```", "").strip()
            
            # Try to parse JSON
            import re
            match = re.search(r'\{.+\}', clean, re.DOTALL)
            if match:
                return json.loads(match.group())
            else:
                # Fallback: try to parse the whole response
                return json.loads(clean)
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
        # Degraded state: model unavailable — defer processing, don't persist
        result = PerceptionResult(
            category="noise",
            salience=0.1,
            action="defer",
            reasoning=f"Model unavailable — deferred: {str(e)}",
            raw_input=data,
            timestamp=datetime.now().isoformat()
        ).to_dict()
        result["status"] = "deferred"
        result["persisted"] = False
        return result

    # Extract and validate fields
    category = model_output.get("category", "noise")
    if category not in CATEGORIES:
        category = "noise"

    salience = float(model_output.get("salience", 0.5))
    salience = max(0.0, min(1.0, salience))  # Clamp to [0, 1]

    # Use model's action if provided, otherwise determine from salience
    action = model_output.get("action", "")
    if action not in ["persist", "discard", "escalate", "defer"]:
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
    out["status"] = "active"

    # Persist to ZenBrain if salience is high enough and not in deferred state
    if persist and salience >= SALIENCE_MEDIUM_THRESHOLD and out.get("status") != "deferred":
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
