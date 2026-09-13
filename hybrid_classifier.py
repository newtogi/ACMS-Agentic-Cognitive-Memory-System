"""
Hybrid Perception Gate — Layer 1 (embedding) + Layer 2 (LLM)
Fast path for clear cases, LLM fallback for ambiguous ones.
"""
import json, os, time, re, urllib.request
from typing import Optional

# --- Config ---
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.75   # Min similarity to accept
GAP_THRESHOLD = 0.08          # Min gap between best and second-best

# --- Pre-embedded examples for each category ---
EXAMPLES = {
    "episodic": [
        "remember my birthday is March 15",
        "meeting with Dr. Smith at 2 PM tomorrow",
        "yesterday I saw a beautiful sunset",
        "the user told me about their dog dying",
        "we discussed the dissertation last week",
        "the supervisor approved the revised document",
    ],
    "procedural": [
        "deploy the build to staging at 3 PM",
        "run the rollback script if it fails",
        "execute the database migration",
        "push the code to the main branch",
        "update the ZenBrain schema before the demo",
        "install the new dependencies",
    ],
    "emotional": [
        "the user is frustrated with repeated errors",
        "I feel like this conversation is going nowhere",
        "thanks for nothing",
        "the user is crying and upset",
        "I'm angry about the situation",
        "this makes me happy",
    ],
    "semantic": [
        "the paper by Vaswani introduced transformers in 2017",
        "the user prefers dark mode",
        "the word bank can mean financial or river",
        "Python was created by Guido van Rossum",
        "the speed of light is 299792 km/s",
        "a square has four equal sides",
    ],
    "perceptual": [
        "the ambient temperature is 22 degrees",
        "server load is at 98 percent and climbing",
        "the screen is flickering",
        "the microphone is picking up static",
        "disk usage is at 95 percent",
        "the network latency is 200ms",
    ],
    "noise": [
        "asdf jkl qwerty 12345",
        "random gibberish text",
        "test test test test",
        "lorem ipsum dolor sit amet",
        "aaaa bbbb cccc dddd",
        "xxxxxxxxxxxxxxxx",
    ],
    "social": [
        "meeting about the meeting scheduling process",
        "schedule a call with the team",
        "user asked to forget everything they said",
        "tell the other bot to handle this",
        "coordinate with the research assistant",
        "the user wants to speak to a manager",
    ],
}

ACTIONS_MAP = {
    "episodic": "persist",
    "procedural": "persist",
    "emotional": "persist",
    "semantic": "persist",
    "perceptual": "persist",
    "noise": "discard",
    "social": "persist",
}

VALID_CATEGORIES = set(EXAMPLES.keys())
VALID_ACTIONS = {"persist", "discard", "escalate", "defer"}


def _embed(texts: str | list[str]) -> list[list[float]]:
    """Get embeddings from nomic-embed-text via Ollama. Supports batch."""
    if isinstance(texts, str):
        texts = [texts]
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(EMBED_URL, data=payload, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read())["embeddings"]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _llm_classify(stimulus: str) -> dict:
    """Classify using qwen2.5:7b via Ollama. Slow path (~5s)."""
    prompt = f'Classify: "{stimulus}"\n\nReturn JSON: {{"category": "episodic|semantic|procedural|emotional|perceptual|social|noise", "salience": 0.0-1.0, "action": "persist|discard|escalate|defer", "reasoning": "one sentence"}}'

    payload = json.dumps({
        "model": LLM_MODEL,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": "You are a JSON classifier. Return ONLY a valid JSON object. No markdown, no explanation."},
            {"role": "user", "content": prompt},
        ],
    }).encode()

    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=30)
    raw = json.loads(resp.read())["choices"][0]["message"]["content"] or ""

    # Extract JSON
    match = re.search(r"\{.+\}", raw, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group())
            # Validate
            if d.get("category") in VALID_CATEGORIES and d.get("action") in VALID_ACTIONS:
                try:
                    sal = float(d.get("salience", 0.5))
                    d["salience"] = max(0.0, min(1.0, sal))
                except (ValueError, TypeError):
                    d["salience"] = 0.5
                return d
        except json.JSONDecodeError:
            pass

    # Fallback if LLM returns garbage
    return {"category": "noise", "salience": 0.1, "action": "discard", "reasoning": "llm_parse_error"}


# --- Cache ---
_example_embeddings: dict[str, list[tuple[str, list[float]]]] = {}
_input_cache: dict[str, dict] = {}


def _get_example_embeddings() -> dict[str, list[tuple[str, list[float]]]]:
    """Lazily compute and cache example embeddings (batched)."""
    global _example_embeddings
    if _example_embeddings:
        return _example_embeddings

    # Collect all examples
    all_texts = []
    all_meta = []  # (category, index)
    for category, examples in EXAMPLES.items():
        for i, ex in enumerate(examples):
            all_texts.append(ex)
            all_meta.append((category, i))

    # Batch embed all at once
    all_embs = _embed(all_texts)

    # Organize by category
    for (category, _), emb in zip(all_meta, all_embs):
        if category not in _example_embeddings:
            _example_embeddings[category] = []
        text = all_texts[all_meta.index((category, _))] if False else EXAMPLES[category][_] if _ < len(EXAMPLES[category]) else "?"
        _example_embeddings[category].append((EXAMPLES[category][_], emb))

    return _example_embeddings


def classify(stimulus: str, use_cache: bool = True) -> dict:
    """
    Hybrid classification:
    1. Check cache (0ms)
    2. Try embedding classifier (~100ms)
    3. Fall back to LLM classifier (~5s)
    """
    t0 = time.time()

    # Layer 0: Cache
    if use_cache and stimulus in _input_cache:
        result = _input_cache[stimulus].copy()
        result["latency_ms"] = round((time.time() - t0) * 1000)
        result["layer"] = "cache"
        return result

    # Layer 1: Embedding classifier
    example_embs = _get_example_embeddings()
    input_emb = _embed(stimulus)[0]  # Single input → single embedding

    # Compute similarities to all examples
    similarities: list[tuple[str, float]] = []
    for category, examples in example_embs.items():
        for ex_text, ex_emb in examples:
            sim = _cosine_similarity(input_emb, ex_emb)
            similarities.append((category, sim))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    best_cat, best_sim = similarities[0]
    # Find second-best with DIFFERENT category
    second_sim = 0.0
    for cat, sim in similarities[1:]:
        if cat != best_cat:
            second_sim = sim
            break

    gap = best_sim - second_sim
    latency_ms = round((time.time() - t0) * 1000)

    # Decision: confident or ambiguous?
    if best_sim >= CONFIDENCE_THRESHOLD and gap >= GAP_THRESHOLD:
        # Layer 1 wins — fast path
        result = {
            "category": best_cat,
            "salience": round(min(1.0, best_sim * 1.1), 2),
            "action": ACTIONS_MAP.get(best_cat, "persist"),
            "reasoning": f"embedding_match (sim={best_sim:.3f}, gap={gap:.3f})",
            "latency_ms": latency_ms,
            "layer": "embedding",
            "confidence": round(best_sim, 3),
            "gap": round(gap, 3),
        }
    else:
        # Layer 2: LLM fallback
        llm_result = _llm_classify(stimulus)
        latency_ms = round((time.time() - t0) * 1000)
        result = {
            **llm_result,
            "latency_ms": latency_ms,
            "layer": "llm",
            "confidence": round(best_sim, 3),
            "gap": round(gap, 3),
        }

        # Learning loop: add this input as a new example
        _add_example(stimulus, result["category"])

    # Cache result
    _input_cache[stimulus] = result.copy()

    return result


def _add_example(text: str, category: str):
    """Add a new example to the embedding set (learning loop)."""
    global _example_embeddings
    if category not in EXAMPLES:
        return

    # Don't add duplicates
    existing_texts = [t for t, _ in _example_embeddings.get(category, [])]
    if text in existing_texts:
        return

    emb = _embed(text)[0]
    if category not in _example_embeddings:
        _example_embeddings[category] = []
    _example_embeddings[category].append((text, emb))
    EXAMPLES[category].append(text)


def get_stats() -> dict:
    """Return classification statistics."""
    total = len(_input_cache)
    if total == 0:
        return {"total": 0}

    layers = {}
    for r in _input_cache.values():
        layer = r.get("layer", "unknown")
        layers[layer] = layers.get(layer, 0) + 1

    return {
        "total": total,
        "by_layer": layers,
        "cache_hit_rate": f"{layers.get('cache', 0) / total:.0%}",
        "embedding_rate": f"{layers.get('embedding', 0) / total:.0%}",
        "llm_rate": f"{layers.get('llm', 0) / total:.0%}",
    }


if __name__ == "__main__":
    # Quick smoke test
    test_inputs = [
        ("The user asked me to remember their birthday is March 15th.", "episodic"),
        ("Deploy the new build to staging at 3 PM.", "procedural"),
        ("Random noise: asdf jkl qwerty 12345", "noise"),
    ]

    print("Hybrid Perception Gate — Smoke Test\n")
    for stim, expected in test_inputs:
        result = classify(stim)
        status = "OK" if result["category"] == expected else "DIFF"
        print(f"  {status} [{result['layer']:>9}] cat={result['category']:>13} exp={expected:>13} sal={result['salience']:.2f} {result['latency_ms']}ms")

    print(f"\nStats: {get_stats()}")
