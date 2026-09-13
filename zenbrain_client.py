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

# Profiles the dream cycle may read. Cross-profile consolidation is opt-in:
# edit this list deliberately — do not leave extra profiles in it "just in case".
# An empty list means single-profile operation (system-bot only).
KNOWN_PROFILES = ["default", "system-bot", "writer", "webhunter"]


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
    """Simple LIKE-based recall across all memory layers.

    Each table has different column names for its content, so the search
    columns vary per table. For vector recall, use the MCP ``zenbrain_recall``
    tool instead — this is the fast SQL fallback for hot loops.
    """
    conn = _connect(profile)
    try:
        pattern = f"%{query}%"
        results: list[dict] = []

        # (table, layer, searchable columns)
        tables = [
            ("episodic_memories", "episodic", ["content"]),
            ("learned_facts", "semantic", ["content"]),
            ("procedural_memories", "procedural", ["trigger", "steps", "outcome"]),
            ("core_memory_blocks", "core", ["content"]),
        ]
        for table, layer, cols in tables:
            where = " OR ".join(f"{c} LIKE ?" for c in cols)
            params = [pattern] * len(cols) + [limit]
            rows = conn.execute(
                f"SELECT id, {cols[0]} AS content FROM {table} WHERE {where} LIMIT ?",
                params,
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
# SCHEMA VALIDATION & BACKUP
# --------------------------------------------------------------------------- #

# Expected schema version — bump when tables/columns change
# Required tables and columns (the schema our code depends on).
# Used by validate_schema() to catch silent corruption or ZenBrain upgrades
# that remove/rename tables we write to.
REQUIRED_SCHEMA: dict[str, list[str]] = {
    "episodic_memories":      ["id", "content", "emotional_weight", "created_at"],
    "learned_facts":          ["id", "content", "confidence", "source", "created_at"],
    "procedural_memories":    ["id", "trigger", "steps", "outcome", "created_at"],
    "core_memory_blocks":     ["id", "label", "content", "pinned"],
    "cross_context_links":    ["id", "entity_a", "entity_b", "created_at"],
}


def validate_schema(profile: str = "system-bot") -> dict[str, list[str]]:
    """
    Check that the expected ZenBrain tables and columns exist.

    Returns a dict mapping each table to its actual columns.
    Raises RuntimeError if any required table or column is missing — this
    protects against silent corruption after a ZenBrain upgrade.
    """
    conn = _connect(profile)
    try:
        missing: list[str] = []
        schema: dict[str, list[str]] = {}
        for table, required_cols in REQUIRED_SCHEMA.items():
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            actual_cols = {r[1] for r in rows}
            schema[table] = sorted(actual_cols)
            for col in required_cols:
                if col not in actual_cols:
                    missing.append(f"{table}.{col}")
        if missing:
            raise RuntimeError(
                f"Schema validation failed for profile '{profile}': "
                f"missing columns: {', '.join(missing)}"
            )
        return schema
    finally:
        conn.close()


def backup_before_write(profile: str = "system-bot") -> Path | None:
    """
    Create a timestamped backup of the database before a write operation.

    Copies zenbrain-{profile}.db to zenbrain-{profile}.{timestamp}.backup
    in the same directory. Returns the backup path, or None if the DB doesn't
    exist yet (first-time setup).
    """
    src = _db_path(profile)
    if not src.exists():
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = src.parent / f"zenbrain-{profile}.{timestamp}.backup"

    import shutil
    shutil.copy2(str(src), str(backup_path))
    return backup_path


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
    backup_before_write(profile)
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
    backup_before_write(profile)
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
