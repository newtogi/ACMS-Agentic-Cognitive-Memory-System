# Agent Brain — Cognitive Architecture Specification

**Version:** 2.0 · **Date:** 2026-09-07 · **Status:** Battle-tested in production use

---

## Who this is for

You are an agent implementing a cognitive memory architecture for an AI system.
This document gives you only what you cannot derive yourself: the design decisions,
the non-obvious integration seams, the schema, and the code worth keeping verbatim.

Everything else — how to install an LLM runtime, how to query a REST API, how to
schedule a task — is within your capability. Negotiate those with your operator as
prompts, not with this document.

**Verification policy:** after each phase, prove the result works before moving on.
If a gate fails, stop and report — do not improvise around a broken foundation.

---

## Credits

| Contribution | Who |
|---|---|
| Architecture, vision, integration | **El** |
| Implementation, testing, documentation | **system-bot** (Hermes Agent, Nous Research) |
| ZenBrain memory engine | Alexander Bering / Zensation AI ([github.com/zensation-ai/zenbrain](https://github.com/zensation-ai/zenbrain), DOI 10.5281/zenodo.19353663) |
| Perception/prediction models | OpenBMB — MiniCPM-V 4.6, MiniCPM5-1B |

---

## 1 · What You Are Building

A **brain** is three cognitive layers wired onto a persistent memory store. It is
deliberately independent of any agent harness — the same brain bolts onto Hermes,
Claude Code, an OpenAI Assistants loop, a raw tool-calling agent, or anything else
that can call an HTTP endpoint and read/write files.

```
                    ┌─────────────────────────────┐
                    │   ANY AGENT HARNESS          │
                    │   (Hermes / Claude / raw)    │
                    └──────────────┬──────────────┘
                                   │ calls brain functions
    ┌──────────────────────────────┼──────────────────────────────┐
    │                AGENT BRAIN (this document)                  │
    │  ┌────────────┐   ┌────────────────┐   ┌────────────────┐  │
    │  │ L1         │   │ L6             │   │ L7             │  │
    │  │ Perception │   │ Predictive     │   │ Dream Cycle    │  │
    │  │ Gate       │   │ Engine         │   │                │  │
    │  └─────┬──────┘   └───────┬────────┘   └───────┬────────┘  │
    │        │                  │                    │           │
    │  ┌─────▼──────────────────▼────────────────────▼────────┐  │
    │  │        ZenBrain — 5 memory layers (SQLite)           │  │
    │  │   episodic · semantic · procedural · core · links    │  │
    │  └──────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
```

The harness's only obligations:

1. Call `perceive(data)` on every significant inbound item, before acting on it.
2. Optionally call `predict(state)` when entering a task, and `record_outcome()`
   when it completes.
3. Trigger `run_dream_cycle()` on a schedule or idle signal.
4. That's it. The brain persists everything through its own client.

---

## 2 · The Layers

| Layer | Function | Contract | Default model |
|---|---|---|---|
| **L1 Perception Gate** | Classify inbound data [event\|fact\|emotion\|threat\|noise], score salience 0–1, recommend [encode\|discard\|escalate]; auto-store above threshold | `perceive(data, profile, persist) -> dict` | small multimodal VLM |
| **L6 Predictive Engine** | Scan episodic patterns, generate a prediction for the current state, estimate confidence, track accuracy vs outcomes | `predict(state) -> dict`, `record_outcome(id, outcome)` | small text LLM |
| **L7 Dream Cycle** | Pull memories across profiles → build NetworkX relationship graph (co-occurrence, temporal, semantic edges) → find cross-domain connections → generate insights → persist them; boost connected memories, decay isolated ones | `run_dream_cycle(live) -> journal entry` | same VLM as L1 |

### Design invariants

- **The gate is not a logger.** If salience < threshold, the item is dropped. A
  brain that stores everything is a log file, not a memory.
- **Predictions must be testable.** Every prediction gets an ID; every outcome
  gets recorded against it; accuracy is a first-class metric, not a vibe.
- **Dreams are the only writer of insights.** Cross-domain synthesis happens
  offline (scheduled/idle), never inline — inline synthesis is just inference.
- **One brain serves every profile.** Layers read/write via profile-scoped
  namespaces against one shared store. No per-bot brain instances.

### Model selection guidance

You need two small local models: one **multimodal** (L1 + L7 — perception must see
images/screenshots) and one **text** (L6 — tool-calling and reasoning). Reference
pair, proven on a 4 GB VRAM laptop with hot-swap serving:

- Multimodal: **MiniCPM-V 4.6** (1.3B, SigLIP2 + Qwen3.5-0.8B, mixed 4×/16× visual token compression)
- Text: **MiniCPM5-1B** (1.08B, LlamaForCausalLM, RL+OPD post-trained, 131K context)

Suitable alternates: SmolVLM2-500M / LFM2-VL-3B / InternVL3-2B for the vision role;
Qwen3-0.6B / Llama3.2-1B / Gemma3-1B for the text role. Requirements: local
serving, ≤ 2B params, JSON-mode capable, vision for the multimodal slot.

**Ask your operator.** Present this table, recommend the reference pair, let them
choose. Write their choice into `config.py`. Nothing else in the brain cares.

---

## 3 · Memory Store: ZenBrain

The memory engine is [ZenBrain](https://github.com/zensation-ai/zenbrain)
(Apache-2.0) — a neuroscience-inspired 7-layer memory system with 20 algorithm
modules. You need its storage layer; the algorithms are worth knowing because the
brain's behavior leans on them.

Clone, install, build per its README. It ships an MCP server
(`packages/mcp/dist/namespace-server.js`); the brain talks to the same SQLite
databases directly (§4) for hot-path speed and falls back to MCP tool calls when
direct access is impossible.

### Schema (v0.4.x — bind to this exactly)

Every profile gets its own DB file: `<db-dir>/zenbrain-<profile>.db`

```sql
episodic_memories     (id TEXT PK, content TEXT NOT NULL, context TEXT,
                       embedding TEXT, emotional_weight REAL, metadata TEXT,
                       created_at TEXT NOT NULL DEFAULT datetime('now'))

learned_facts         (id TEXT PK, content TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 0.7,
                       source TEXT NOT NULL DEFAULT 'conversation', embedding TEXT,
                       access_count INTEGER NOT NULL DEFAULT 0,
                       fsrs_difficulty REAL, fsrs_stability REAL, fsrs_next_review TEXT,
                       created_at TEXT NOT NULL DEFAULT datetime('now'),
                       last_accessed TEXT)

procedural_memories   (id TEXT PK, trigger TEXT, steps TEXT, tools TEXT, outcome TEXT,
                       embedding TEXT, success_rate REAL,
                       execution_count INTEGER, created_at TEXT)

core_memory_blocks    (id TEXT PK, label TEXT, content TEXT, pinned INTEGER, updated_at TEXT)

cross_context_links   (id TEXT PK, entity_a TEXT NOT NULL, entity_b TEXT NOT NULL,
                       created_at TEXT NOT NULL DEFAULT datetime('now'))

knowledge_entities    (id TEXT PK, name TEXT, type TEXT, embedding TEXT, created_at TEXT)
```

WAL mode is on. Two non-negotiables:

- **Never share these files across hosts.** SQLite + WAL does not tolerate
  concurrent multi-host writers; FSRS fields will corrupt. One brain per machine,
  one DB dir per machine.
- **Episodic has no `confidence` column.** Emotional weight plays that role there.
  Don't invent columns — the engine's code owns this schema.

### Where the algorithms live in behavior

| Algorithm (ZenBrain module) | Neuroscience | Where the brain uses it |
|---|---|---|
| `fsrs` | spaced repetition | review scheduling on facts; dream-cycle boost cadence |
| `ebbinghaus` | R = e^(−t/S) decay | episodic decay in dream cycle |
| `hebbian` / `hebbian-two-factor` | fire-together-wire-together | graph edge weights; boost for co-accessed memories |
| `bayesian` | belief propagation | confidence updates when facts corroborate/conflict |
| `sleep-consolidation` | Tononi & Cirelli synaptic renormalization | the dream cycle itself — strengthen + prune |
| `dopamine-routing` | reward prediction error (Schultz 1997) | L6 record_outcome → strengthen confirmed predictions |
| `emotional` | amygdala tagging | emotional_weight drives encode priority and decay resistance |
| `personalized-pagerank` | PageRank | memory importance ranking in the graph |
| `hopfield-stm` | associative recall | partial-cue recall behavior |
| `surprise-gradient-memory` | surprise-modulated encoding | norepinephrine analog — novel items encode stronger |

---

## 4 · The Integration Seams (what actually broke, so you don't have to)

These are the four non-obvious facts discovered during the original build. An
agent would lose hours to each. This is why the client code is given verbatim.

1. **DB location is install-dependent.** ZenBrain DBs may live at `~/.hermes/zenbrain/`
   or `~/AppData/Local/hermes/zenbrain/` (Windows) — and `HERMES_HOME` env may point
   at a *profile* dir, not the root. Resolve by walking up from env, then trying
   known candidates, and **validate by requiring the marker DB file to exist** —
   a bare directory-name match once resolved to an unrelated source repo and
   silently created empty DBs.
2. **Schema drift is real.** The episodic table has no `confidence` column
   (emotional_weight instead); `cross_context_links` uses `entity_a`/`entity_b`
   with its own auto id. Writing the columns you assumed → `OperationalError` at
   the worst moment. Verify with `PRAGMA table_info` before first write.
3. **Small local models will not return clean JSON unless the parser is paranoid.**
   Chain-of-thought, code fences, and preamble text all show up. Parse in order:
   direct JSON → regex-extracted JSON object → structured fallback. Store both the
   clean value and the raw response.
4. **VRAM is a scheduling problem.** A 4 GB GPU cannot hold a 8B general model and
   the brain models simultaneously. Either hot-swap (accept ~2–3 s cold starts,
   tune keep-alive) or run the small models CPU-side (acceptable: 1B inference is
   5–15 tok/s, fine for classification and dreams). Decide with your operator.

---

## 5 · The Brain Code

Nine files, ~2,100 lines, Python 3.11 stdlib only (no pip dependencies; NetworkX
is optional for the graph layer — stdlib fallback exists). Deploy as a package
into `~/agent-brain/`. The code is given verbatim because it encodes §4's lessons.

### 5.1 `config.py`

```python
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
```

### 5.2 `zenbrain_client.py`

The storage client. All layers go through this. Contains the path resolution and
schema handling from §4.

```python
"""
ZenBrain SQLite Client — direct database access for the Agent Brain.

Reads from and writes to the ZenBrain SQLite databases that back each profile.
Uses ONLY the 5-memory-layer schema (episodic, semantic, procedural, core)
plus cross-context links and knowledge entities for the graph builder.

Schema (from ZenBrain v1.x):
  episodic_memories       — time-stamped events with emotional weight
  learned_facts           — semantic facts with confidence
  procedural_memories     — workflows with steps + tools + outcome
  core_memory_blocks      — non-decaying critical info
  cross_context_links     — relationships between memories
  knowledge_entities      — named entities (people, projects, concepts)

This bypasses the MCP server entirely — we talk to SQLite directly because
we're running inside the same Hermes process and the MCP layer would add
overhead to every perception/prediction/dream call.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# Resolve ZenBrain home. Two possible locations depending on Hermes install:
#   ~/.hermes/zenbrain/         — older self-managed install (where MCP points)
#   ~/AppData/Local/hermes/zenbrain/ — newer Windows-native Hermes
# Walk up from HERMES_HOME if set, then try known candidates in priority order.
def _resolve_zenbrain_dir() -> Path:
    # Walk up from HERMES_HOME looking for a parent with a zenbrain/ dir.
    env_home = os.environ.get("HERMES_HOME")
    if env_home:
        parent = Path(env_home).parent
        for _ in range(5):
            candidate = parent / "zenbrain"
            if candidate.exists() and (candidate / "zenbrain-system-bot.db").exists():
                return candidate
            parent = parent.parent

    # Known candidates in priority order. ~/.hermes/zenbrain is where the MCP
    # server is configured (see ZENBRAIN_DB_DIR in profile config.yaml).
    candidates = [
        Path.home() / ".hermes" / "zenbrain",
        Path.home() / "AppData/Local/hermes" / "zenbrain",
    ]
    for c in candidates:
        if c.exists() and (c / "zenbrain-system-bot.db").exists():
            return c

    # Fallback (will not exist; caller will see FileNotFoundError).
    return candidates[0]


ZENBRAIN_DIR = _resolve_zenbrain_dir()

# Profiles that have a ZenBrain database. Add new profiles here as they spin up.
KNOWN_PROFILES = ["default", "system-bot", "writer", "dummy"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path(profile: str) -> Path:
    """Resolve the SQLite path for a given profile."""
    return ZENBRAIN_DIR / f"zenbrain-{profile}.db"


def _connect(profile: str) -> sqlite3.Connection:
    path = _db_path(profile)
    if not path.exists():
        raise FileNotFoundError(f"ZenBrain DB not found for profile '{profile}': {path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------------- #
# READ: pull memories from ZenBrain
# --------------------------------------------------------------------------- #

def get_all_episodes(profile: str = "system-bot", limit: int = 500) -> list[dict]:
    """Pull recent episodic memories for a single profile."""
    conn = _connect(profile)
    try:
        rows = conn.execute(
            """
            SELECT id, content, context, emotional_weight, metadata, created_at
            FROM episodic_memories
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_episode(r) for r in rows]
    finally:
        conn.close()


def get_all_episodes_cross_profile(limit_per_profile: int = 200) -> list[dict]:
    """Pull episodic memories from EVERY known profile. Used by the dream cycle."""
    all_episodes: list[dict] = []
    for profile in KNOWN_PROFILES:
        try:
            all_episodes.extend(get_all_episodes(profile, limit_per_profile))
        except FileNotFoundError:
            continue
    return all_episodes


def get_all_facts(profile: str = "system-bot", limit: int = 200) -> list[dict]:
    """Pull semantic facts for a profile."""
    conn = _connect(profile)
    try:
        rows = conn.execute(
            """
            SELECT id, content, confidence, created_at, source
            FROM learned_facts
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "content": r["content"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
                "source": r["source"],
                "layer": "semantic",
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_all_facts_cross_profile(limit_per_profile: int = 100) -> list[dict]:
    """Pull semantic facts from every profile. Used by the dream cycle."""
    all_facts: list[dict] = []
    for profile in KNOWN_PROFILES:
        try:
            all_facts.extend(get_all_facts(profile, limit_per_profile))
        except FileNotFoundError:
            continue
    return all_facts


def get_cross_context_links(profile: str = "system-bot") -> list[dict]:
    """Pull existing cross-context links for a profile."""
    conn = _connect(profile)
    try:
        rows = conn.execute(
            "SELECT id, entity_a, entity_b, created_at FROM cross_context_links"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def recall(query: str, profile: str = "system-bot", limit: int = 10) -> list[dict]:
    """Simple LIKE-based recall across episodic + semantic + procedural layers.

    For vector recall, use the MCP `zenbrain_recall` tool. This is the
    quick SQL fallback for the Agent Brain's hot loops.
    """
    conn = _connect(profile)
    try:
        pattern = f"%{query}%"
        results: list[dict] = []

        for table, layer in (
            ("episodic_memories", "episodic"),
            ("learned_facts", "semantic"),
            ("procedural_memories", "procedural"),
            ("core_memory_blocks", "core"),
        ):
            rows = conn.execute(
                f"SELECT id, content FROM {table} WHERE content LIKE ? LIMIT ?",
                (pattern, limit),
            ).fetchall()
            for r in rows:
                results.append({"id": r["id"], "content": r["content"], "layer": layer})

        return results[:limit]
    finally:
        conn.close()


def _row_to_episode(row: sqlite3.Row) -> dict:
    """Normalise an episodic_memories row to a flat dict."""
    metadata: dict = {}
    if row["metadata"]:
        try:
            metadata = json.loads(row["metadata"])
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    return {
        "id": row["id"],
        "content": row["content"],
        "context": row["context"] or "",
        "emotional_weight": row["emotional_weight"] or 0.0,
        "created_at": row["created_at"],
        "layer": "episodic",
        "metadata": metadata,
    }


# --------------------------------------------------------------------------- #
# WRITE: store new memories
# --------------------------------------------------------------------------- #

def store_episode(
    content: str,
    profile: str = "system-bot",
    *,
    context: str = "",
    emotional_weight: float = 0.0,
    metadata: dict | None = None,
) -> str:
    """Store an episodic memory. Returns the new memory ID."""
    memory_id = str(uuid.uuid4())
    conn = _connect(profile)
    try:
        conn.execute(
            """
            INSERT INTO episodic_memories
                (id, content, context, emotional_weight, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                content,
                context,
                emotional_weight,
                json.dumps(metadata or {}),
                _now_iso(),
            ),
        )
        conn.commit()
        return memory_id
    finally:
        conn.close()


def store_fact(
    content: str,
    profile: str = "system-bot",
    *,
    confidence: float = 0.8,
    source: str = "agent_brain",
) -> str:
    """Store a semantic fact. Returns the new memory ID."""
    memory_id = str(uuid.uuid4())
    conn = _connect(profile)
    try:
        conn.execute(
            """
            INSERT INTO learned_facts (id, content, confidence, created_at, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (memory_id, content, confidence, _now_iso(), source),
        )
        conn.commit()
        return memory_id
    finally:
        conn.close()


def store_cross_link(
    entity_a: str,
    entity_b: str,
    profile: str = "system-bot",
) -> str:
    """Record a cross-context link between two memory entities. Used by the dream cycle."""
    link_id = str(uuid.uuid4())
    conn = _connect(profile)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO cross_context_links
                (id, entity_a, entity_b, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (link_id, entity_a, entity_b, _now_iso()),
        )
        conn.commit()
        return link_id
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Health / diagnostics
# --------------------------------------------------------------------------- #

def list_profiles() -> list[dict]:
    """List all known profiles with DB presence and basic counts."""
    out: list[dict] = []
    for profile in KNOWN_PROFILES:
        path = _db_path(profile)
        if not path.exists():
            continue
        try:
            conn = _connect(profile)
            try:
                counts = {}
                for table in (
                    "episodic_memories",
                    "learned_facts",
                    "procedural_memories",
                    "core_memory_blocks",
                    "cross_context_links",
                ):
                    cur = conn.execute(f"SELECT COUNT(*) AS n FROM {table}")
                    counts[table] = cur.fetchone()["n"]
                out.append({"profile": profile, "counts": counts, "path": str(path)})
            finally:
                conn.close()
        except Exception as exc:
            out.append({"profile": profile, "error": str(exc)})
    return out


if __name__ == "__main__":
    # Quick smoke test when run directly
    print("=== ZenBrain Client ===")
    print(f"Home: {ZENBRAIN_DIR}")
    print()
    print("Profiles found:")
    for entry in list_profiles():
        if "error" in entry:
            print(f"  {entry['profile']}: ERROR — {entry['error']}")
            continue
        c = entry["counts"]
        print(
            f"  {entry['profile']}: "
            f"{c['episodic_memories']} episodes, "
            f"{c['learned_facts']} facts, "
            f"{c['procedural_memories']} procedures, "
            f"{c['core_memory_blocks']} core, "
            f"{c['cross_context_links']} links"
        )
```

### 5.3 `perception_gate.py`

```python
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
```

### 5.4 `pattern_detector.py`

```python
"""
Pattern Detector — Sequence extraction for Agent Brain L6 Predictive Engine.

Scans episodic memory for recurring sequences, extracts cause-effect pairs,
and computes pattern confidence based on frequency and recency.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Episode normalisation
# ---------------------------------------------------------------------------

def _parse_ts(ts: str | float | None) -> datetime | None:
    """Best-effort timestamp parser (ISO-8601 or epoch seconds)."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(ts), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _normalise_episode(ep: dict) -> dict:
    """Return a copy with parsed timestamp and cleaned content."""
    return {
        "content": str(ep.get("content", "")).strip(),
        "timestamp": _parse_ts(ep.get("timestamp")),
        "layer": ep.get("layer", "unknown"),
    }


# ---------------------------------------------------------------------------
# Sequence fingerprint
# ---------------------------------------------------------------------------

def _fingerprint(*tokens: str) -> str:
    """Stable short hash for a token sequence."""
    return hashlib.sha1("||".join(tokens).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PatternDetector:
    """Detects recurring sequences and cause-effect pairs in episodic memory."""

    def __init__(
        self,
        episodes: list[dict],
        *,
        min_support: int = 2,
        max_ngram: int = 4,
        recency_half_life_days: float = 30.0,
    ):
        self.episodes: list[dict] = [_normalise_episode(e) for e in episodes]
        self.min_support = min_support
        self.max_ngram = max_ngram
        self.recency_half_life_days = recency_half_life_days

        # Internal caches populated on first scan
        self._sequences: list[dict] | None = None
        self._cause_effects: list[dict] | None = None

    # ----- public helpers -----

    def scan_sequences(self) -> list[dict]:
        """
        Detect recurring n-gram sequences across episodes (ordered by timestamp).

        Returns a list of dicts:
            {
                "id":          str,            # fingerprint
                "tokens":      list[str],      # the repeated token sequence
                "count":       int,            # how many times it appears
                "confidence":  float,          # 0-1 weighted score
                "layers":      list[str],      # layers where it appeared
                "last_seen":   datetime | None # most recent occurrence
            }
        """
        if self._sequences is not None:
            return self._sequences

        ordered = sorted(
            [e for e in self.episodes if e["timestamp"]],
            key=lambda e: e["timestamp"],
        )
        texts = [e["content"] for e in ordered]
        layers_per_ep = [e["layer"] for e in ordered]

        # Tokenise: split on whitespace / punctuation boundaries
        def tokenize(text: str) -> list[str]:
            return [t.lower() for t in re.findall(r"\w+", text)]

        tokenised = [(tokenize(t), layers_per_ep[i]) for i, t in enumerate(texts)]

        # Count every 2..max_ngram contiguous subsequence across episodes
        ngram_counter: Counter[tuple[str, ...]] = Counter()
        ngram_layers: dict[tuple[str, ...], set[str]] = defaultdict(set)
        ngram_last_idx: dict[tuple[str, ...], int] = {}

        for ep_idx, (tokens, layer) in enumerate(tokenised):
            for n in range(2, min(self.max_ngram + 1, len(tokens) + 1)):
                for start in range(len(tokens) - n + 1):
                    gram = tuple(tokens[start : start + n])
                    ngram_counter[gram] += 1
                    ngram_layers[gram].add(layer)
                    ngram_last_idx[gram] = ep_idx

        # Also detect *non-contiguous* recurring keyword sequences (skip-grams)
        for ep_idx, (tokens, layer) in enumerate(tokenised):
            for n in range(2, min(self.max_ngram + 1, len(tokens) + 1)):
                # Only do skip-grams for bigrams to keep it tractable
                if n == 2 and len(tokens) >= 3:
                    for i in range(len(tokens)):
                        for j in range(i + 2, min(i + 5, len(tokens))):
                            gram = (tokens[i], tokens[j])
                            ngram_counter[gram] += 1
                            ngram_layers[gram].add(layer)
                            ngram_last_idx[gram] = ep_idx

        # Filter by min_support and score
        now = datetime.now(timezone.utc)
        results: list[dict] = []
        for gram, count in ngram_counter.items():
            if count < self.min_support:
                continue
            last_ep_idx = ngram_last_idx.get(gram, 0)
            last_ts = ordered[last_ep_idx]["timestamp"] if ordered else None
            confidence = self._compute_confidence(count, last_ts, now)
            results.append({
                "id": _fingerprint(*gram),
                "tokens": list(gram),
                "count": count,
                "confidence": round(confidence, 4),
                "layers": sorted(ngram_layers.get(gram, set())),
                "last_seen": last_ts.isoformat() if last_ts else None,
            })

        results.sort(key=lambda r: r["confidence"], reverse=True)
        self._sequences = results
        return results

    def extract_cause_effect_pairs(self) -> list[dict]:
        """
        Extract cause→effect pairs from temporally adjacent episodes.

        A "pair" is two consecutive episodes (by timestamp) whose content
        satisfies a simple heuristic:
            cause  = episode with action/event keywords ("because", "due to",
                     "triggered", "if", "when", "caused")
            effect = the immediately following episode

        Returns:
            {
                "id":         str,
                "cause":      str,          # cause text
                "effect":     str,          # effect text
                "cause_ep_idx": int,
                "effect_ep_idx": int,
                "gap_seconds": float | None,
                "count":      int,          # how many times this pair re-occurred
                "confidence": float,
                "layers":     list[str],
            }
        """
        if self._cause_effects is not None:
            return self._cause_effects

        CAUSE_SIGNALS = re.compile(
            r"\b(because|due\s+to|triggered|caused|if\s+\w|when\s+\w|after\s+|"
            r"since|resulted|led\s+to|reason)\b",
            re.IGNORECASE,
        )

        ordered = sorted(
            [e for e in self.episodes if e["timestamp"]],
            key=lambda e: e["timestamp"],
        )

        pair_counter: Counter[tuple[str, str]] = Counter()
        pair_meta: dict[tuple[str, str], dict] = {}

        for i in range(len(ordered) - 1):
            curr = ordered[i]
            nxt = ordered[i + 1]

            # Heuristic: current episode mentions a cause signal, or simply
            # treat *any* sequential pair as a potential cause-effect pair
            cause_text = curr["content"]
            effect_text = nxt["content"]

            if not cause_text or not effect_text:
                continue

            # Truncate for fingerprint stability
            cause_short = cause_text[:200]
            effect_short = effect_text[:200]
            key = (cause_short, effect_short)

            pair_counter[key] += 1
            if key not in pair_meta:
                gap = None
                if curr["timestamp"] and nxt["timestamp"]:
                    gap = (nxt["timestamp"] - curr["timestamp"]).total_seconds()
                pair_meta[key] = {
                    "cause_ep_idx": i,
                    "effect_ep_idx": i + 1,
                    "gap_seconds": gap,
                    "layers": set(),
                }
            pair_meta[key]["layers"].add(curr["layer"])
            pair_meta[key]["layers"].add(nxt["layer"])

        now = datetime.now(timezone.utc)
        results: list[dict] = []
        for (cause, effect), count in pair_counter.items():
            meta = pair_meta[(cause, effect)]
            last_ts = ordered[meta["effect_ep_idx"]]["timestamp"]
            confidence = self._compute_confidence(count, last_ts, now)
            # Boost confidence for explicit cause-signal text
            if CAUSE_SIGNALS.search(cause):
                confidence = min(1.0, confidence * 1.3)

            results.append({
                "id": _fingerprint(cause[:60], effect[:60]),
                "cause": cause,
                "effect": effect,
                "cause_ep_idx": meta["cause_ep_idx"],
                "effect_ep_idx": meta["effect_ep_idx"],
                "gap_seconds": meta["gap_seconds"],
                "count": count,
                "confidence": round(confidence, 4),
                "layers": sorted(meta["layers"]),
            })

        results.sort(key=lambda r: r["confidence"], reverse=True)
        self._cause_effects = results
        return results

    # ----- internals -----

    def _compute_confidence(
        self,
        count: int,
        last_seen: datetime | None,
        now: datetime,
    ) -> float:
        """
        Confidence = freq_score × recency_score.

        freq_score    = log2(count+1) / log2(max_support+1)   ∈ (0, 1]
        recency_score = 0.5 ^ (age_days / half_life)          ∈ (0, 1]
        """
        max_count = max((s["count"] for s in self._sequences or []), default=count)
        max_count = max(max_count, count, 1)

        freq_score = math.log2(count + 1) / math.log2(max_count + 1)

        if last_seen is None:
            recency_score = 0.3  # penalise unknown recency
        else:
            age_days = max((now - last_seen).total_seconds() / 86400, 0)
            recency_score = math.pow(0.5, age_days / self.recency_half_life_days)

        return max(0.0, min(1.0, freq_score * recency_score))

    def summarise(self) -> str:
        """Return a compact text summary of top patterns for prompt injection."""
        seqs = self.scan_sequences()[:15]
        ces = self.extract_cause_effect_pairs()[:10]

        parts: list[str] = []
        if seqs:
            parts.append("RECURRING PATTERNS:")
            for s in seqs:
                parts.append(
                    f"  [{' '.join(s['tokens'])}] ×{s['count']} "
                    f"conf={s['confidence']:.2f} last={s.get('last_seen','?')}"
                )
        if ces:
            parts.append("CAUSE→EFFECT PAIRS:")
            for c in ces:
                parts.append(
                    f"  [{c['cause'][:80]}…] → [{c['effect'][:80]}…] "
                    f"×{c['count']} conf={c['confidence']:.2f}"
                )
        return "\n".join(parts) if parts else "(no patterns detected)"
```

### 5.5 `predictive_engine.py`

```python
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
```

### 5.6 `memory_graph.py`

```python
"""
memory_graph.py — NetworkX graph operations for the Agent Brain Dream Cycle.

Nodes = memories (with attributes: id, content, layer, context, confidence,
                       emotional_weight, strength, timestamp)
Edges = relationships (type: co_occurrence | temporal | semantic, weight: float)
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import networkx as nx


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    id: str
    content: str
    layer: str  # episodic, semantic, procedural, core
    context: str = ""
    confidence: float = 0.5
    emotional_weight: float = 0.0
    strength: float = 1.0
    timestamp: str = ""  # ISO 8601
    tags: list[str] = field(default_factory=list)
    access_count: int = 0


# ---------------------------------------------------------------------------
# Token helpers (lightweight — no external NLP dependency)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "the a an and or but in on at to for of is it that this with from by as are was were be been".split()
)


def _tokenize(text: str) -> set[str]:
    """Lowercase alpha-numeric tokens, stop-words removed."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _parse_ts(iso: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.fromisoformat(iso.rstrip("Z"))
        except (ValueError, AttributeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class MemoryGraph:
    """Undirected weighted graph over agent memories."""

    def __init__(self):
        self.G: nx.Graph = nx.Graph()
        self._token_cache: dict[str, set[str]] = {}

    # -- population ---------------------------------------------------------

    def add_memory(self, mem: Memory) -> None:
        self.G.add_node(
            mem.id,
            content=mem.content,
            layer=mem.layer,
            context=mem.context,
            confidence=mem.confidence,
            emotional_weight=mem.emotional_weight,
            strength=mem.strength,
            timestamp=mem.timestamp,
            tags=mem.tags,
            access_count=mem.access_count,
        )
        self._token_cache[mem.id] = _tokenize(mem.content + " " + mem.context)

    def build_from_memories(self, memories: list[Memory]) -> None:
        for m in memories:
            self.add_memory(m)

    # -- edge creation ------------------------------------------------------

    def connect_co_occurrence(self, threshold: float = 0.25) -> int:
        """Edge when two memories share enough tokens (Jaccard)."""
        added = 0
        ids = list(self.G.nodes)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                sim = _jaccard(self._token_cache.get(a, set()), self._token_cache.get(b, set()))
                if sim >= threshold:
                    self.G.add_edge(a, b, relation="co_occurrence", weight=sim)
                    added += 1
        return added

    def connect_temporal(self, hours: float = 24.0, max_edges: int = 200) -> int:
        """Edge when two memories were created within `hours` of each other.
        Weight decays exponentially with time gap."""
        added = 0
        nodes_with_ts: list[tuple[str, datetime]] = []
        for n, data in self.G.nodes(data=True):
            ts = _parse_ts(data.get("timestamp", ""))
            if ts:
                nodes_with_ts.append((n, ts))
        nodes_with_ts.sort(key=lambda x: x[1])

        tau = timedelta(hours=hours)
        for i in range(len(nodes_with_ts)):
            if added >= max_edges:
                break
            for j in range(i + 1, len(nodes_with_ts)):
                gap = nodes_with_ts[j][1] - nodes_with_ts[i][1]
                if gap > tau:
                    break
                w = math.exp(-gap.total_seconds() / tau.total_seconds())
                self.G.add_edge(nodes_with_ts[i][0], nodes_with_ts[j][0],
                                relation="temporal", weight=round(w, 4))
                added += 1
        return added

    def connect_semantic(self, threshold: float = 0.15) -> int:
        """Edge when memories share context domain or tags (lightweight semantic)."""
        added = 0
        ids = list(self.G.nodes)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                da = self.G.nodes[ids[i]]
                db = self.G.nodes[ids[j]]
                score = 0.0
                # same context domain
                if da.get("context") and da["context"] == db.get("context"):
                    score += 0.4
                # shared tags
                ta, tb = set(da.get("tags", [])), set(db.get("tags", []))
                if ta and tb:
                    score += 0.6 * len(ta & tb) / len(ta | tb) if (ta | tb) else 0
                if score >= threshold:
                    self.G.add_edge(ids[i], ids[j], relation="semantic", weight=round(score, 4))
                    added += 1
        return added

    def build_all_edges(self, co_thresh=0.25, temporal_hours=24.0, semantic_thresh=0.15) -> dict[str, int]:
        c = self.connect_co_occurrence(co_thresh)
        t = self.connect_temporal(temporal_hours)
        s = self.connect_semantic(semantic_thresh)
        return {"co_occurrence": c, "temporal": t, "semantic": s}

    # -- analytics ----------------------------------------------------------

    def clusters(self) -> list[set[str]]:
        """Return connected components as list of node-id sets."""
        return [comp for comp in nx.connected_components(self.G)]

    def cluster_subgraphs(self) -> list[nx.Graph]:
        return [self.G.subgraph(c).copy() for c in nx.connected_components(self.G)]

    def top_connected(self, n: int = 10) -> list[tuple[str, int]]:
        """Nodes ranked by degree."""
        return sorted(self.G.degree(), key=lambda x: x[1], reverse=True)[:n]

    def isolated_nodes(self) -> list[str]:
        return [n for n, d in self.G.degree() if d == 0]

    def bridge_edges(self) -> list[tuple[str, str]]:
        """Edges whose removal would disconnect the graph (cross-domain links)."""
        return list(nx.bridges(self.G)) if nx.is_connected(self.G) else []

    def node_centrality(self) -> dict[str, float]:
        return nx.betweenness_centrality(self.G)

    def summary(self) -> dict[str, Any]:
        """Compact summary for inclusion in LLM prompts."""
        return {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "num_clusters": len(self.clusters()),
            "isolated_count": len(self.isolated_nodes()),
            "avg_degree": round(sum(dict(self.G.degree()).values()) / max(self.G.number_of_nodes(), 1), 2),
            "top_nodes": [(n, d) for n, d in self.top_connected(5)],
        }

    # -- strength adjustment -----------------------------------------------

    def adjust_strengths(self, boost_factor: float = 0.1, decay_factor: float = 0.05) -> dict[str, float]:
        """Boost well-connected nodes, decay isolated ones. Returns delta map."""
        degree_map = dict(self.G.degree())
        max_deg = max(degree_map.values()) if degree_map else 1
        deltas: dict[str, float] = {}
        for n, data in self.G.nodes(data=True):
            deg = degree_map.get(n, 0)
            old = data.get("strength", 1.0)
            if deg == 0:
                new = max(0.0, old - decay_factor)
            else:
                # boost proportional to normalized degree
                new = min(1.0, old + boost_factor * (deg / max_deg))
            self.G.nodes[n]["strength"] = round(new, 4)
            deltas[n] = round(new - old, 4)
        return deltas

    # -- export -------------------------------------------------------------

    def to_dict(self) -> dict:
        return nx.node_link_data(self.G)

    def cluster_summaries(self, max_per_cluster: int = 5) -> list[dict]:
        """Return one summary dict per cluster (for LLM cross-domain analysis)."""
        summaries = []
        for comp in nx.connected_components(self.G):
            sub = self.G.subgraph(comp)
            nodes = []
            for n, d in sub.nodes(data=True):
                nodes.append({
                    "id": n,
                    "content": d.get("content", "")[:200],
                    "layer": d.get("layer", ""),
                    "context": d.get("context", ""),
                    "strength": d.get("strength", 1.0),
                })
            # prioritise by strength
            nodes.sort(key=lambda x: x["strength"], reverse=True)
            summaries.append({
                "size": len(nodes),
                "contexts": list({n["context"] for n in nodes if n["context"]}),
                "sample": nodes[:max_per_cluster],
            })
        return summaries
```

### 5.7 `dream_cycle.py`

```python
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
```

### 5.8 `__init__.py`

```python
"""
Agent Brain - L1 Perception Gate Module

A modular perception layer that classifies incoming data and determines
salience for downstream processing by the agent brain.
"""

from .perception_gate import perceive, PerceptionResult

__all__ = ["perceive", "PerceptionResult"]
```

### 5.9 `requirements.txt`

```
# Agent Brain - L1 Perception Gate Dependencies
# Uses Ollama HTTP API directly - no additional Python packages required
# All dependencies are Python standard library (urllib, json)

# Optional: for async support
# aiohttp>=3.9.0

# Optional: for structured output validation
# pydantic>=2.0.0
```

---

## 6 · Wiring Into Your Harness

The integration surface is four calls. How your harness triggers them is harness
policy — negotiate with your operator (hook points, which events count as
"significant", idle detection). For reference, the Hermes wiring:

| Trigger | Call | Effect |
|---|---|---|
| Inbound message/tool result | `perceive(data)` | classified + auto-stored if salient |
| Task start | `predict(state)` | prediction ID returned for later scoring |
| Task completion | `record_outcome(id, outcome)` | dopamine analog: accuracy tracked |
| Scheduled (default 02:00) or idle | `run_dream_cycle(live=True)` | graph → connections → insights → persisted facts |

Dream insights land in `learned_facts` with `source='agent_brain_dream_cycle'`,
so **every profile on the machine can recall them immediately** through whatever
recall mechanism the harness already has. That is the entire cross-bot benefit —
no per-bot wiring required.

---

## 7 · Acceptance Test

Run after deployment. All steps must pass:

```
1. perceive("User received feedback from supervisor about the response document")
   → category=event, salience ≥ 0.4, and a memory_id present
2. SELECT COUNT(*) FROM episodic_memories WHERE id = <memory_id>  → 1
3. run_dream_cycle(live=True)  → journal entry with insights; log shows
   "Persisting insights to ZenBrain … stored insight"
4. SELECT COUNT(*) FROM learned_facts WHERE source='agent_brain_dream_cycle'  → ≥ 1
5. predict(<current state>)  → prediction is a clean short string (< 200 chars),
   no chain-of-thought, JSON-extracted
```

Failure at any step → see §8.

---

## 8 · Known Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: ZenBrain DB not found` | path resolution missed the install layout | check `zenbrain_client._resolve_zenbrain_dir()`; verify marker DB exists |
| `table episodic_memories has no column named confidence` | wrote assumed schema | schema is §3 — emotional_weight, not confidence |
| model returns prose/CoT instead of JSON | small-model JSON discipline | `_parse_prediction` / `_extract_json` fallback chain must be present |
| `HTTP 404` from model endpoint | model name mismatch vs serving registry | config model name must equal the served tag exactly |
| everything classified as noise + "Model unavailable" | serving runtime down | restart the local model server, re-run acceptance test |
| dream cycle finds 0 connections | memories too few or graph too sparse | needs ≥ ~10 memories with temporal/semantic overlap to be meaningful |

---

## Papers

- Ebbinghaus, H. (1885). *Über das Gedächtnis* — forgetting curve, R = e^(−t/S)
- Hebb, D. O. (1949). *The Organization of Behavior* — fire together, wire together
- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS 79*(8)
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems* — Bayesian confidence propagation
- Schultz, W. et al. (1997). A Neural Substrate of Prediction and Reward. *Science 275* — dopamine reward prediction error
- Tononi, G. & Cirelli, C. (2014). Sleep and the Price of Plasticity. *Neuron 81*(1) — synaptic renormalization during sleep
- Page, L. et al. (1999). The PageRank Citation Ranking — memory importance
- Open Spaced Repetition — [FSRS](https://github.com/open-spaced-repetition/fsrs4anki)
- Codebook Agent (2026). Amortized Topology Design for LLM Multi-Agent Systems. arXiv:2609.02264 — agent-communication topology
- Selective Forgetting (2026). Graph-based memory pruning. arXiv:2608.28978 — importance = 0.35·recency + 0.25·frequency + 0.20·centrality + 0.20·decay
- OpenBMB — MiniCPM-V 4.6 (LLaVA-UHD v4 intra-ViT compression); MiniCPM5-1B (RL + On-Policy Distillation)
- Bering, A. (2026). ZenBrain: A Neuroscience-Inspired 7-Layer Memory Architecture for Autonomous AI Systems. arXiv:2604.23878
