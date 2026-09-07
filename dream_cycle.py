"""
dream_cycle.py — Agent Brain L7 Dream Cycle

Orchestrates a "sleep" pass over stored memories:
  1. Retrieve all memories from ZenBrain (mock or live via MCP).
  2. Build a NetworkX relationship graph.
  3. Ask MiniCPM-V 4.6 (Ollama) to find cross-domain connections.
  4. Generate insights from those connections.
  5. Adjust memory strengths (boost connected, decay isolated).
  6. Write a dream journal entry to ~/agent-brain/dream_journal.json.

Run directly:  python dream_cycle.py
Programmatic:  from dream_cycle import run_dream_cycle; run_dream_cycle()
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import requests

# local import — sibling module
from memory_graph import Memory, MemoryGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DREAM] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dream_cycle")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AGENT_BRAIN_DIR = Path(__file__).resolve().parent
JOURNAL_PATH = AGENT_BRAIN_DIR / "dream_journal.json"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "minicpm-v4.6:latest")

# ZenBrain MCP is invoked through hermes_tools when available;
# standalone runs fall back to the mock memories below.


# ===================================================================
# 1. MEMORY RETRIEVAL
# ===================================================================

MOCK_MEMORIES: list[dict[str, Any]] = [
    {
        "id": "mem-001", "content": "Python 3.12 adds type-parameter syntax PEP 695",
        "layer": "semantic", "context": "coding", "confidence": 0.9,
        "emotional_weight": 0.1, "strength": 0.85,
        "timestamp": "2026-09-01T10:00:00Z", "tags": ["python", "typing"],
    },
    {
        "id": "mem-002", "content": "User prefers dark mode in all applications",
        "layer": "semantic", "context": "personal", "confidence": 0.95,
        "emotional_weight": 0.2, "strength": 0.9,
        "timestamp": "2026-08-15T14:30:00Z", "tags": ["preference", "ui"],
    },
    {
        "id": "mem-003", "content": "Deployed staging environment for project-alpha using Docker Compose",
        "layer": "episodic", "context": "work", "confidence": 0.85,
        "emotional_weight": 0.4, "strength": 0.7,
        "timestamp": "2026-09-03T09:15:00Z", "tags": ["docker", "deploy", "staging"],
    },
    {
        "id": "mem-004", "content": "NetworkX is great for building knowledge graphs from unstructured data",
        "layer": "semantic", "context": "coding", "confidence": 0.8,
        "emotional_weight": 0.1, "strength": 0.75,
        "timestamp": "2026-09-05T16:00:00Z", "tags": ["networkx", "graph", "python"],
    },
    {
        "id": "mem-005", "content": "Consolidation in ZenBrain promotes repeated episodes to semantic facts",
        "layer": "semantic", "context": "ai-research", "confidence": 0.88,
        "emotional_weight": 0.3, "strength": 0.8,
        "timestamp": "2026-09-06T11:00:00Z", "tags": ["zenbrain", "memory", "consolidation"],
    },
    {
        "id": "mem-006", "content": "Procedure: run pytest -x --tb=short for fast test feedback",
        "layer": "procedural", "context": "coding", "confidence": 0.95,
        "emotional_weight": 0.05, "strength": 0.9,
        "timestamp": "2026-08-20T08:00:00Z", "tags": ["testing", "pytest", "workflow"],
    },
    {
        "id": "mem-007", "content": "The user's birthday is in October and they like astronomy",
        "layer": "core", "context": "personal", "confidence": 0.99,
        "emotional_weight": 0.8, "strength": 0.95,
        "timestamp": "2026-07-01T12:00:00Z", "tags": ["birthday", "astronomy", "personal"],
    },
    {
        "id": "mem-008", "content": "MiniCPM-V supports multimodal reasoning on images and text via Ollama",
        "layer": "semantic", "context": "ai-research", "confidence": 0.85,
        "emotional_weight": 0.2, "strength": 0.7,
        "timestamp": "2026-09-07T08:00:00Z", "tags": ["minicpm", "ollama", "multimodal"],
    },
    {
        "id": "mem-009", "content": "User asked to build an L7 dream cycle for memory consolidation",
        "layer": "episodic", "context": "work", "confidence": 0.95,
        "emotional_weight": 0.5, "strength": 1.0,
        "timestamp": "2026-09-07T14:00:00Z", "tags": ["dream-cycle", "agent-brain", "project"],
    },
    {
        "id": "mem-010", "content": "Sleep replay in neuroscience strengthens hippocampal-cortical traces",
        "layer": "semantic", "context": "ai-research", "confidence": 0.75,
        "emotional_weight": 0.3, "strength": 0.6,
        "timestamp": "2026-09-04T20:00:00Z", "tags": ["neuroscience", "sleep", "memory"],
    },
    {
        "id": "mem-011", "content": "Failed CI build was caused by missing .env variable DATABASE_URL",
        "layer": "episodic", "context": "work", "confidence": 0.9,
        "emotional_weight": 0.6, "strength": 0.65,
        "timestamp": "2026-09-02T17:30:00Z", "tags": ["ci", "debugging", "env"],
    },
    {
        "id": "mem-012", "content": "Heroku to Railway migration saved 40% on hosting costs",
        "layer": "episodic", "context": "work", "confidence": 0.88,
        "emotional_weight": 0.5, "strength": 0.7,
        "timestamp": "2026-08-28T10:00:00Z", "tags": ["heroku", "railway", "cost"],
    },
]


def retrieve_memories_mock() -> list[Memory]:
    """Convert mock dicts into Memory objects."""
    return [Memory(**m) for m in MOCK_MEMORIES]


def retrieve_memories_zenbrain(query: str = "*", limit: int = 100) -> list[Memory]:
    """Live recall from ZenBrain via the SQLite client.

    Pulls both episodic and semantic memories from all known profiles,
    ranked by recency. This is the Agent Brain's primary memory source.
    """
    try:
        from zenbrain_client import (
            get_all_episodes_cross_profile,
            get_all_facts_cross_profile,
        )

        episodes = get_all_episodes_cross_profile(limit_per_profile=limit // 4)
        facts = get_all_facts_cross_profile(limit_per_profile=limit // 4)

        # Interleave and convert to Memory objects
        raw: list[dict] = []
        for ep in episodes:
            raw.append(
                {
                    "id": ep["id"],
                    "content": ep["content"],
                    "layer": "episodic",
                    "confidence": ep["confidence"],
                    "emotionalWeight": ep["emotional_weight"],
                    "timestamp": ep["created_at"],
                }
            )
        for f in facts:
            raw.append(
                {
                    "id": f["id"],
                    "content": f["content"],
                    "layer": "semantic",
                    "confidence": f["confidence"],
                    "timestamp": f["created_at"],
                }
            )

        # Apply a simple LIKE filter if query isn't the wildcard
        if query and query != "*":
            ql = query.lower()
            raw = [r for r in raw if ql in r["content"].lower()]

        # Sort by timestamp desc, take top N
        raw.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        raw = raw[:limit]

        if not raw:
            log.warning("ZenBrain returned no memories, falling back to mock")
            return retrieve_memories_mock()

        memories: list[Memory] = []
        for r in raw:
            memories.append(
                Memory(
                    id=r["id"],
                    content=r["content"],
                    layer=r.get("layer", "semantic"),
                    confidence=r.get("confidence", 0.5),
                    emotional_weight=r.get("emotionalWeight", 0.0),
                    timestamp=r.get("timestamp", ""),
                )
            )
        return memories
    except Exception as e:
        log.warning("ZenBrain recall unavailable (%s), falling back to mock", e)
        return retrieve_memories_mock()


# ===================================================================
# 3. OLLAMA / MiniCPM-V 4.6
# ===================================================================

def _ollama_generate(prompt: str, system: str = "", temperature: float = 0.7,
                     max_tokens: int = 1024) -> str:
    """Call Ollama /api/chat and return the response text."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "format": "json",
    }

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    except requests.RequestException as e:
        log.error("Ollama call failed: %s", e)
        return f"[OLLAMA_ERROR: {e}]"


def _extract_json(text: str) -> dict | None:
    """Robustly extract JSON from LLM response (handles markdown fences)."""
    # Try markdown code block first
    import re
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else [text]
    for raw in candidates:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def find_cross_domain_connections(cluster_summaries: list[dict]) -> dict[str, Any]:
    """Ask MiniCPM-V to identify novel connections between memory clusters."""
    system = (
        "You are an AI memory consolidation agent. Your job is to find surprising, "
        "non-obvious connections between different clusters of memories. These connections "
        "represent cross-domain insights that could be valuable for future reasoning. "
        "Always respond with valid JSON only — no markdown fences, no commentary."
    )
    prompt = f"""Analyze these memory clusters and identify cross-domain connections.

MEMORY CLUSTERS:
{json.dumps(cluster_summaries, indent=2)}

For each connection you find, provide:
1. The memory IDs involved
2. The type of connection (analogy, causal, temporal_pattern, contradiction, reinforcement, gap)
3. A brief explanation of the connection
4. A novelty score (0-1)

Return ONLY valid JSON — no markdown fences:
{{
  "connections": [
    {{"memories": ["id1", "id2"], "type": "...", "explanation": "...", "novelty": 0.0}}
  ],
  "meta_reflection": "Brief reflection on what these connections reveal"
}}"""

    raw = _ollama_generate(prompt, system=system, temperature=0.8, max_tokens=2048)
    log.debug("Raw connections response: %s", raw[:500])
    parsed = _extract_json(raw)
    if parsed:
        return parsed
    # Last resort: try to interpret the raw text as a connection list
    return {"connections": [], "raw_response": raw[:1000], "meta_reflection": "Parse failed — see raw_response"}


def generate_insights(connections_data: dict[str, Any], graph_summary: dict) -> list[dict]:
    """Produce actionable insights from cross-domain connections."""
    system = (
        "You are a memory consolidation agent generating actionable insights. "
        "Each insight should suggest something the agent can DO or REMEMBER differently."
    )
    prompt = f"""Given these cross-domain connections and graph statistics, generate 3-5 actionable insights.

CONNECTIONS:
{json.dumps(connections_data, indent=2)}

GRAPH SUMMARY:
{json.dumps(graph_summary, indent=2)}

Return JSON:
{{
  "insights": [
    {{"title": "...", "detail": "...", "action": "...", "priority": "high|medium|low"}}
  ]
}}"""

    raw = _ollama_generate(prompt, system=system, temperature=0.6, max_tokens=2048)
    parsed = _extract_json(raw)
    if parsed and "insights" in parsed:
        insights = parsed["insights"]
        # Handle case where model returns insights as a JSON string
        if isinstance(insights, str):
            inner = _extract_json(insights)
            if inner and "insights" in inner:
                insights = inner["insights"]
            else:
                insights = [{"title": "Parsed from text", "detail": insights[:500], "action": "review", "priority": "medium"}]
        return insights
    return [{"title": "Insight generation", "detail": raw[:500], "action": "review", "priority": "low"}]


# ===================================================================
# 5. MEMORY STRENGTH ADJUSTMENT (via ZenBrain or local graph)
# ===================================================================

def adjust_strengths(graph: MemoryGraph) -> dict[str, float]:
    """Boost connected memories, decay isolated ones."""
    deltas = graph.adjust_strengths(boost_factor=0.1, decay_factor=0.05)
    log.info("Strength adjustments: %d nodes modified, %d boosted, %d decayed",
             len(deltas),
             sum(1 for v in deltas.values() if v > 0),
             sum(1 for v in deltas.values() if v < 0))
    return deltas


# ===================================================================
# 6. DREAM JOURNAL
# ===================================================================

def load_journal() -> list[dict]:
    if JOURNAL_PATH.exists():
        try:
            return json.loads(JOURNAL_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def write_journal_entry(
    graph: MemoryGraph,
    connections: dict[str, Any],
    insights: list[dict],
    strength_deltas: dict[str, float],
    edge_stats: dict[str, int],
) -> dict:
    """Append a new dream entry to the journal."""
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": now,
        "model": OLLAMA_MODEL,
        "graph_snapshot": graph.summary(),
        "edge_creation": edge_stats,
        "cross_domain_connections": connections.get("connections", []),
        "meta_reflection": connections.get("meta_reflection", ""),
        "insights": insights,
        "strength_deltas": {
            "boosted": {k: v for k, v in strength_deltas.items() if v > 0},
            "decayed": {k: v for k, v in strength_deltas.items() if v < 0},
            "unchanged": sum(1 for v in strength_deltas.values() if v == 0),
        },
    }

    journal = load_journal()
    journal.append(entry)
    JOURNAL_PATH.write_text(json.dumps(journal, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Dream journal entry written: %s (total entries: %d)", entry["id"], len(journal))
    return entry


# ===================================================================
# 7. MAIN ORCHESTRATOR
# ===================================================================

def run_dream_cycle(use_mock: bool = True) -> dict:
    """Execute a full L7 dream cycle and return the journal entry.

    Args:
        use_mock: If True, use mock memories. If False, attempt ZenBrain recall.

    Returns:
        The dream journal entry dict.
    """
    log.info("═══════════════════════════════════════════")
    log.info("  L7 DREAM CYCLE — starting")
    log.info("═══════════════════════════════════════════")

    # Step 1: Retrieve memories
    log.info("[1/6] Retrieving memories …")
    if use_mock:
        memories = retrieve_memories_mock()
    else:
        memories = retrieve_memories_zenbrain()
    log.info("  → %d memories loaded", len(memories))

    # Step 2: Build graph
    log.info("[2/6] Building memory relationship graph …")
    graph = MemoryGraph()
    graph.build_from_memories(memories)
    edge_stats = graph.build_all_edges()
    summary = graph.summary()
    log.info("  → Graph: %d nodes, %d edges, %d clusters",
             summary["num_nodes"], summary["num_edges"], summary["num_clusters"])
    log.info("  → Edge breakdown: %s", edge_stats)

    # Step 3: Cross-domain analysis via LLM
    log.info("[3/6] Finding cross-domain connections (model: %s) …", OLLAMA_MODEL)
    cluster_sums = graph.cluster_summaries()
    connections = find_cross_domain_connections(cluster_sums)
    num_conn = len(connections.get("connections", []))
    log.info("  → %d connections found", num_conn)
    if connections.get("meta_reflection"):
        log.info("  → Meta: %s", connections["meta_reflection"][:200])

    # Step 4: Generate insights
    log.info("[4/6] Generating insights …")
    insights = generate_insights(connections, summary)
    log.info("  → %d insights generated", len(insights))

    # Step 5: Adjust memory strengths
    log.info("[5/6] Adjusting memory strengths …")
    strength_deltas = adjust_strengths(graph)

    # Step 6: Write dream journal
    log.info("[6/6] Writing dream journal …")
    entry = write_journal_entry(graph, connections, insights, strength_deltas, edge_stats)

    # Step 7: Persist insights back to ZenBrain so all bots can recall them
    log.info("[7/7] Persisting insights to ZenBrain …")
    if not use_mock:
        try:
            from zenbrain_client import store_fact
            for insight in insights:
                title = insight.get("title", "Untitled insight")
                detail = insight.get("detail", "")
                content = f"[DREAM INSIGHT] {title}: {detail}"
                mem_id = store_fact(
                    content=content,
                    profile="system-bot",
                    confidence=0.85,
                    source="agent_brain_dream_cycle",
                )
                log.info("  → stored insight: %s", mem_id[:8])
        except Exception as exc:
            log.warning("Could not persist insights to ZenBrain: %s", exc)

    log.info("═══════════════════════════════════════════")
    log.info("  L7 DREAM CYCLE — complete")
    log.info("═══════════════════════════════════════════")

    return entry


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    mock = "--live" not in sys.argv
    result = run_dream_cycle(use_mock=mock)
    print(json.dumps(result, indent=2, ensure_ascii=False))
