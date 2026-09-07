"""
Agent Brain L1 Perception Gate Configuration
Model names and thresholds for the perception layer.
"""

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "minicpm-v4.6"

# Perception categories
CATEGORIES = ["event", "fact", "emotion", "threat", "noise"]

# Salience thresholds
SALIENCE_HIGH_THRESHOLD = 0.7    # Above this: escalate
SALIENCE_MEDIUM_THRESHOLD = 0.4  # Above this: encode
SALIENCE_LOW_THRESHOLD = 0.2     # Below this: discard

# Action mapping based on salience
ACTION_THRESHOLDS = {
    "escalate": SALIENCE_HIGH_THRESHOLD,
    "encode": SALIENCE_MEDIUM_THRESHOLD,
    "discard": SALIENCE_LOW_THRESHOLD
}

# JSON output schema for the model
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": CATEGORIES
        },
        "salience": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "reasoning": {
            "type": "string"
        }
    },
    "required": ["category", "salience", "reasoning"]
}
