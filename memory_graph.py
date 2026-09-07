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
