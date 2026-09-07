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
