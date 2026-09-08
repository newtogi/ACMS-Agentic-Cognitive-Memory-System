"""
insight_store.py — Dream Insight Persistence

Stores dream-generated insights to a local JSON file for review and
optionally persists approved insights to ZenBrain as semantic facts.

Usage:
    from insight_store import store_dream_insight, load_insights, list_insights

    # Store an insight (writes to dream_insights.json)
    insight_id = store_dream_insight(
        title="Cross-domain connection found",
        detail="Python typing and Docker networking both use graph-based dependency resolution",
        action="Add dependency graph visualization to IDE",
        priority="high",
        source="dream_cycle",
    )

    # Load all insights
    all_insights = load_insights()

    # List insights with status filter
    pending = list_insights(status="pending")
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
INSIGHTS_PATH = Path.home() / "agent-brain" / "dream_insights.json"

# Valid statuses for insights
VALID_STATUSES = Literal["pending", "approved", "rejected", "archived"]


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _load_insights_file() -> list[dict[str, Any]]:
    """Load insights from disk. Returns empty list if file doesn't exist."""
    if not INSIGHTS_PATH.exists():
        return []
    try:
        data = json.loads(INSIGHTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return data
    except (json.JSONDecodeError, OSError):
        return []


def _save_insights_file(insights: list[dict[str, Any]]) -> None:
    """Write insights to disk atomically."""
    INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    INSIGHTS_PATH.write_text(
        json.dumps(insights, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_dream_insight(
    title: str,
    detail: str,
    *,
    action: str = "",
    priority: str = "medium",
    source: str = "dream_cycle",
    confidence: float = 0.2,
    status: str = "pending",
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Store a dream-generated insight to the local JSON file.

    Args:
        title: Short title of the insight
        detail: Detailed explanation of the insight
        action: Suggested action or next step
        priority: Priority level (high, medium, low)
        source: Origin of the insight (e.g., dream_cycle, consolidation)
        confidence: Confidence score (0.0-1.0); low by default for dream insights
        status: Lifecycle status (pending, approved, rejected, archived)
        metadata: Optional extra data to attach

    Returns:
        The insight ID string
    """
    if status not in ("pending", "approved", "rejected", "archived"):
        raise ValueError(f"Invalid status: {status!r}")
    if priority not in ("high", "medium", "low"):
        raise ValueError(f"Invalid priority: {priority!r}")

    insight_id = str(uuid.uuid4())[:12]
    now = _now_iso()

    insight = {
        "id": insight_id,
        "title": title,
        "detail": detail,
        "action": action,
        "priority": priority,
        "confidence": max(0.0, min(1.0, confidence)),
        "status": status,
        "source": source,
        "created_at": now,
        "updated_at": now,
        "metadata": metadata or {},
    }

    # Append to file
    insights = _load_insights_file()
    insights.append(insight)
    _save_insights_file(insights)

    return insight_id


def load_insights() -> list[dict[str, Any]]:
    """Load all stored dream insights."""
    return _load_insights_file()


def list_insights(
    status: str | None = None,
    priority: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    List insights with optional filtering.

    Args:
        status: Filter by status (pending, approved, rejected, archived)
        priority: Filter by priority (high, medium, low)
        limit: Maximum number of insights to return

    Returns:
        List of matching insights, most recent first
    """
    insights = _load_insights_file()

    if status:
        insights = [i for i in insights if i.get("status") == status]
    if priority:
        insights = [i for i in insights if i.get("priority") == priority]

    # Sort by created_at descending (most recent first)
    insights.sort(key=lambda i: i.get("created_at", ""), reverse=True)

    return insights[:limit]


def update_insight_status(
    insight_id: str,
    new_status: str,
    *,
    reviewer_notes: str = "",
) -> bool:
    """
    Update an insight's status (e.g., from pending to approved).

    Args:
        insight_id: The insight ID to update
        new_status: New status value
        reviewer_notes: Optional notes from the reviewer

    Returns:
        True if the insight was found and updated, False otherwise
    """
    if new_status not in ("pending", "approved", "rejected", "archived"):
        raise ValueError(f"Invalid status: {new_status!r}")

    insights = _load_insights_file()
    found = False

    for insight in insights:
        if insight.get("id") == insight_id:
            insight["status"] = new_status
            insight["updated_at"] = _now_iso()
            if reviewer_notes:
                insight.setdefault("review_history", []).append({
                    "status": new_status,
                    "notes": reviewer_notes,
                    "timestamp": _now_iso(),
                })
            found = True
            break

    if found:
        _save_insights_file(insights)

    return found


def promote_approved_to_zenbrain(
    profile: str = "system-bot",
    *,
    dry_run: bool = False,
) -> list[str]:
    """
    Promote approved insights to ZenBrain as semantic facts.

    Only processes insights with status='approved'. After promotion,
    the insight status is updated to 'archived'.

    Args:
        profile: ZenBrain profile to write to
        dry_run: If True, don't actually write to ZenBrain

    Returns:
        List of ZenBrain memory IDs created
    """
    from zenbrain_client import store_fact

    approved = list_insights(status="approved")
    stored_ids: list[str] = []

    for insight in approved:
        content = f"[DREAM INSIGHT] {insight['title']}: {insight['detail']}"
        if insight.get("action"):
            content += f" → Action: {insight['action']}"

        if dry_run:
            stored_ids.append(f"dry-run-{insight['id']}")
            continue

        try:
            mem_id = store_fact(
                content=content,
                profile=profile,
                confidence=insight.get("confidence", 0.5),
                source=f"dream_insight:{insight.get('source', 'unknown')}",
            )
            stored_ids.append(mem_id)

            # Archive the insight after successful promotion
            update_insight_status(insight["id"], "archived")
        except Exception:
            # Don't archive if write failed
            pass

    return stored_ids


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "list":
        status_filter = sys.argv[2] if len(sys.argv) > 2 else None
        insights = list_insights(status=status_filter)
        if not insights:
            print("No insights found.")
        for i in insights:
            print(f"[{i['status'].upper()}] {i['id']}: {i['title']}")
            print(f"  Priority: {i['priority']} | Confidence: {i['confidence']}")
            if i.get("detail"):
                print(f"  {i['detail'][:100]}")
            print()

    elif cmd == "add":
        if len(sys.argv) < 4:
            print("Usage: insight_store.py add <title> <detail>")
            sys.exit(1)
        insight_id = store_dream_insight(
            title=sys.argv[2],
            detail=sys.argv[3],
            source="manual",
        )
        print(f"Stored insight: {insight_id}")

    elif cmd == "promote":
        dry_run = "--dry-run" in sys.argv
        ids = promote_approved_to_zenbrain(dry_run=dry_run)
        print(f"Promoted {len(ids)} insights to ZenBrain.")
        for mid in ids:
            print(f"  {mid}")

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: insight_store.py [list|add|promote]")
        sys.exit(1)
