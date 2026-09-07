# Agent Brain

**A cognitive memory architecture for AI agents.** Three brain layers — perception, prediction, dream consolidation — wired onto [ZenBrain](https://github.com/zensation-ai/zenbrain), a neuroscience-inspired memory store. Harness-agnostic: bolts onto Hermes, Claude Code, a raw tool loop, or anything that can call a function. Fully local. Zero API cost.

[![License: MIT](https://img.shields.io/badge/License-MIT-22d3ee.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.11+-a78bfa.svg)](https://www.python.org)
[![ZenBrain](https://img.shields.io/badge/Memory-ZenBrain_0.4.x-34d399.svg)](https://github.com/zensation-ai/zenbrain)
[![Models](https://img.shields.io/badge/Models-MiniCPM--V_4.6_·_MiniCPM5--1B-fbbf24.svg)](https://huggingface.co/OpenBMB)

---

## Why

LLM agents forget. Context windows reset, sessions end, and every conversation starts from zero. Retrieval is not memory — a RAG lookup returns text, but nothing decides *what deserves to be remembered*, nothing *anticipates*, and nothing *consolidates* while the agent is idle.

Agent Brain adds those three missing pieces:

| Missing piece | What Agent Brain does |
|---|---|
| **Filtering** | An LLM agent that stores everything is a log file, not a mind. L1 scores every input for salience and drops the noise before it reaches memory. |
| **Anticipation** | L6 scans episodic history for recurring sequences and predicts what happens next — then scores itself against reality. Accuracy is a metric, not a vibe. |
| **Consolidation** | Brains don't learn in real time; they consolidate offline. L7 runs while you sleep: builds a graph of everything remembered, finds cross-domain connections nobody noticed, writes insights back for every profile to recall in the morning. |

The result: an agent that gets sharper over time without any weight updates — because the intelligence lives in what it keeps, what it expects, and what it connects.

## Architecture

```
                    ┌─────────────────────────────┐
                    │      ANY AGENT HARNESS       │
                    │  Hermes · Claude · raw loop  │
                    └──────────────┬──────────────┘
                                   │  4 calls — that's the whole contract
    ┌──────────────────────────────┼──────────────────────────────┐
    │                AGENT BRAIN (one per machine)                │
    │                                                             │
    │   L1 Perception ──► L6 Predictive ──► L7 Dream Cycle        │
    │   Gate              Engine              (2 AM)              │
    │                                                             │
    └──────────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  ZENBRAIN MEMORY      │
                    │  episodic · semantic  │
                    │  procedural · core    │
                    │  cross-links          │
                    └──────────────────────┘
```

**The entire integration surface is 4 calls:**

```python
from perception_gate import perceive        # → category, salience, action, memory_id
from predictive_engine import PredictiveEngine
engine = PredictiveEngine()
engine.predict(state)                       # → prediction_id, prediction, confidence
engine.record_outcome(pid, outcome)         # → confirmed / refuted + accuracy delta
from dream_cycle import run_dream_cycle
run_dream_cycle(live=True)                  # → graph stats, connections, insights
```

Triggers, hooks, and schedules are **harness policy** — wire them however your agent works. Dream insights land in ZenBrain's `learned_facts` with a `source='agent_brain_dream_cycle'` tag, so every profile on the machine recalls them immediately through whatever recall mechanism it already has. One brain, every bot — no per-bot wiring.

## The Layers

| Layer | Contract | What it enforces | Model |
|---|---|---|---|
| **L1 Perception Gate** | `perceive(data) → dict` | Classifies [event\|fact\|emotion\|threat\|noise], scores salience 0–1, recommends [encode\|discard\|escalate]. **The gate is not a logger** — below threshold, the input is dropped. | Local VLM, 1–2B |
| **L6 Predictive Engine** | `predict(state)` · `record_outcome(id, outcome)` | Scans episodic sequences, predicts the next outcome with confidence. **Predictions must be testable** — every one gets an ID, and outcomes score against it. | Local text LLM, 1–2B |
| **L7 Dream Cycle** | `run_dream_cycle(live) → journal` | Offline synthesis, **never inline**. Builds a NetworkX graph (co-occurrence, temporal, semantic edges), finds cross-domain connections, generates insights, boosts connected memories, decays isolated ones. | Same VLM as L1 |

### The learning loop

```
👁 perceive → 💾 store → 🔗 connect → 🔮 predict → 🌙 dream → 💡 insight → back to 👁
```

Insights feed back into perception — the loop closes. That is the whole mechanism by which the system "learns" without touching model weights.

### Under the hood — ZenBrain

The memory store isn't a vector DB with a name. ZenBrain implements actual cognitive-science algorithms, and the brain's behavior leans on them:

| Module | Science | Where it shows up |
|---|---|---|
| `fsrs` | spaced repetition | fact review scheduling, dream-cycle boost cadence |
| `ebbinghaus` | R = e^(−t/S) decay | episodic forgetting in the dream cycle |
| `hebbian` | fire together, wire together | graph edge weights, co-access boosts |
| `bayesian` | belief propagation | confidence updates on corroborate/conflict |
| `sleep-consolidation` | synaptic renormalization | the dream cycle itself |
| `dopamine-routing` | reward prediction error | confirmed predictions strengthen |
| `emotional` | amygdala tagging | `emotional_weight` drives priority + decay resistance |
| `personalized-pagerank` | PageRank | memory importance in the graph |
| `hopfield-stm` | associative recall | partial-cue recall |
| `surprise-gradient-memory` | surprise-modulated encoding | novel items encode stronger |

## Models

Two small local models. Reference pair, proven on a 4 GB VRAM laptop:

| Role | Reference | Alternatives |
|---|---|---|
| Multimodal (L1 + L7) — must see images | [MiniCPM-V 4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) · 1.3B | SmolVLM2-500M · LFM2-VL-3B · InternVL3-2B |
| Text (L6) — JSON + tools | [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) · 1.08B | Qwen3-0.6B · Llama3.2-1B · Gemma3-1B |

**Requirements:** local serving (Ollama, llama.cpp, vLLM — anything OpenAI-compatible) · ≤ 2B params · JSON-mode capable · vision for the multimodal slot. Model names are config, not code — swap freely.

**VRAM is a scheduling problem.** A 4 GB GPU can't hold a large general model and the brain models at once. Either hot-swap (accept ~2–3 s cold starts) or run the 1B models CPU-side — 5–15 tok/s is plenty for classification and dream synthesis.

## Quick Start

Prerequisites you almost certainly have: Python 3.11+, a local model server with two small models loaded, and [ZenBrain](https://github.com/zensation-ai/zenbrain) built (its README covers clone → `npm install` → build; the MCP server lands at `packages/mcp/dist/namespace-server.js`).

```bash
git clone <this-repo> ~/agent-brain
cd ~/agent-brain

# Point config.py at your models and memory store, then:
python -c "import sys; sys.path.insert(0, '.'); from perception_gate import perceive; print(perceive('hello world'))"
```

> **Note:** `config.py` ships with the reference model names and auto-resolves the ZenBrain DB location across known install layouts. Both are one-line edits.

Full machine-readable deployment spec — including the exact SQLite schema, the integration seams that cost us hours, and all code inline — is in [`agent-brain-spec.md`](agent-brain-spec.md). A visual overview is in [`agent-brain-infographic.html`](agent-brain-infographic.html).

### Wiring it up (any harness)

| When | Call |
|---|---|
| Significant inbound item arrives | `perceive(data)` — before acting on it |
| Task starts | `engine.predict(state)` — keep the ID |
| Task completes | `engine.record_outcome(pid, outcome)` |
| Schedule or idle (default: 02:00) | `run_dream_cycle(live=True)` |

## Design Invariants

1. **The gate is not a logger.** A brain that stores everything is a log file. Salience threshold is the point.
2. **Predictions must be testable.** No prediction without an ID; no ID without an outcome check.
3. **Dreams are the only insight writer.** Cross-domain synthesis happens offline. Inline "synthesis" is just inference.
4. **One brain serves every profile.** Profile-scoped namespaces against one shared store — never per-bot brain instances.
5. **Never share the SQLite files across hosts.** WAL mode + multi-host writers = FSRS corruption. One brain per machine.

## Honest Limitations

- **The dream cycle needs ~10+ memories with overlap** before cross-domain connections become meaningful. Cold start is thin.
- **Each machine's brain is autonomous.** Insights don't auto-propagate across machines — that's deliberate (SQLite won't survive it), and if you need it, sync at the application layer.
- **Small models misbehave.** Expect chain-of-thought leakage; the parsers are paranoid for a reason.
- **L1 is only as good as its VLM.** Sub-2B vision models hallucinate on dense screenshots. The salience threshold absorbs some of this; not all.
- **No weight updates, ever.** "Learning" = better memory selectivity + sharper pattern prediction. The model itself is frozen.

## Project Structure

```
agent-brain/
├── config.py               # model names, salience thresholds, JSON schema
├── zenbrain_client.py      # SQLite client — path resolution, schema-safe reads/writes
├── perception_gate.py      # L1 — classify + salience + auto-store
├── pattern_detector.py     # L6 — sequence mining, cause-effect extraction
├── predictive_engine.py    # L6 — predict + outcome scoring + accuracy stats
├── memory_graph.py         # L7 — NetworkX relationship graph
├── dream_cycle.py          # L7 — synthesis orchestration + journal
├── LICENSE                 # MIT
├── agent-brain-spec.md     # machine-readable deployment spec (for agents)
├── agent-brain-infographic.html
└── requirements.txt        # networkx + requests (L7); stdlib otherwise
```

## Credits

| | |
|---|---|
| **Architecture, vision, integration** | El |
| **Implementation, testing, documentation** | system-bot (Hermes Agent, Nous Research) |
| **Memory engine** | [ZenBrain](https://github.com/zensation-ai/zenbrain) — Alexander Bering / Zensation AI · arXiv:2604.23878 · DOI 10.5281/zenodo.19353663 |
| **Reference models** | [OpenBMB](https://huggingface.co/OpenBMB) — MiniCPM-V 4.6 (LLaVA-UHD v4), MiniCPM5-1B (RL + OPD) |
| **Framework** | [Hermes Agent](https://hermes-agent.nousresearch.com) — Nous Research |

### Scientific foundations

Ebbinghaus (1885) *Über das Gedächtnis* · Hebb (1949) *The Organization of Behavior* · Hopfield (1982) *PNAS 79*(8) · Pearl (1988) *Probabilistic Reasoning in Intelligent Systems* · Schultz et al. (1997) *Science 275* · Tononi & Cirelli (2014) *Neuron 81*(1) · Page et al. (1999) PageRank · [FSRS](https://github.com/open-spaced-repetition/fsrs4anki) · Codebook Agent (arXiv:2609.02264) · Selective Forgetting (arXiv:2608.28978)

## License

Brain code: MIT. ZenBrain: Apache-2.0 (per its repo). Model licenses per their model cards.
