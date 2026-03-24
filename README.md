# omega-session

**Ω-gated KV eviction for bounded-memory LLM sessions on consumer hardware.**

A session governance method that maintains semantic coherence over arbitrarily long conversations within a fixed token budget. A structural coherence score (Ω) computed against a session axis declared at t=0 governs eviction of context entries — preserving semantically load-bearing content while displacing topically orthogonal noise.

Runs locally. No cloud. No external ML dependencies. Works with any Ollama model.

---

## Results (Qwen2.5-14B, Apple M3, 32GB, 20 turns)

| Metric | Governed | Baseline |
|--------|----------|----------|
| Throughput | **37.5 t/s** | 27.7 t/s |
| Speedup | **+35.3%** | — |
| Budget violations | **0 / 20 turns** | n/a |
| Early recall (planted fact) | **recalled** | lost |
| Final eviction rate | **57.3%** (51 of 89 units) | n/a |

The +35.3% throughput gain comes from bounded prefill cost: the governed system feeds a fixed token budget to the model each turn regardless of session length, while naive baseline prefill grows linearly.

---

## How It Works

```
Session open
    |
    v
axis = sha256(first_user_message)   <- hash-committed at t=0, never updated
    |
    v  each turn
Parse history -> ContextUnits        <- argument AST segmentation
    |
    v
Score each unit:
    overlap = |axis_words ∩ unit_words| / |unit_words|
    structural_Ω = 1.0 - min(overlap × 3.0, 1.0)
    composite = (1 - Ω) × recency_weight × role_weight × solvency
    |
    v
Pin latest user turn (causality invariant)
Greedily select highest-composite units within token budget
Evict the rest for this turn (retained in local store, re-scored next turn)
    |
    v
Feed governed window -> Ollama -> response
```

The axis is the governing question of the session. Every context unit is scored as deviation from that question — not from the current conversational state, not from a sliding average. Units that drift far from the axis are evicted first. Units that stay coherent with the origin survive longest.

---

## Install

Requires Python 3.9+ and [Ollama](https://ollama.ai) running locally.

```bash
git clone https://github.com/YOUR_USERNAME/omega-session
cd omega-session

# No pip install needed — stdlib only
# Pull your model if you haven't already
ollama pull qwen2.5:14b
```

---

## Usage

### Interactive session

```bash
python omega_session.py --model qwen2.5:14b --budget 3000 --verbose
```

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen2.5:14b` | Any Ollama model |
| `--budget` | `3000` | Max tokens fed per turn |
| `--verbose` | off | Show governance stats per turn |

Commands during session: `report` prints session stats, `quit` exits.

### Run the test harness

```bash
# Unit tests only (no Ollama needed)
python test_harness.py --skip-live

# Full suite, 8 turns (~3 min)
python test_harness.py --model qwen2.5:14b --turns 8

# Stress run, 20 turns (full eviction curves, ~30 min)
python test_harness.py --model qwen2.5:14b --turns 20
```

Six tests:

| Test | Needs Ollama | Measures |
|------|-------------|----------|
| T1 | No | Segmentation correctness, switch detection |
| T2 | No | Ω scoring direction, eviction, pin invariant |
| T3 | Yes | t/s stability — governed vs naive baseline |
| T4 | Yes | Budget never violated across N turns |
| T5 | Yes | Early recall: planted fact vs sliding window |
| T6 | Yes | Eviction curves over full session |

---

## Files

```
omega_session.py    <- Entry point. CLI + Ollama client + session loop.
stp_session.py      <- Transport layer. Axis declaration, Ω scorer,
                       window builder, append-only audit log.
context_stack.py    <- Ingestion layer. Argument AST segmenter,
                       ContextUnit dataclass, IEP window formula.
test_harness.py     <- Systematic test suite (T1-T6).
```

No other dependencies. Everything imports upward from `context_stack.py`.

---

## Governance Invariants

Five structural invariants enforced at every turn:

- **I1 Causality** — governance precedes the model call, always
- **I2 Conservation** — token budget is finite, tracked, enforced as hard ceiling
- **I3 Separation** — the scorer is architecturally separate from the generator
- **I4 Irreversibility** — the session audit log is append-only
- **I5 Identity** — the axis is declared at t=0, hash-committed, never updated

---

## Known Limitations (v0)

**Lexical scorer.** The Ω scorer uses word-overlap against the axis text. This works for natural conversation but fails for technically dense writing where semantic proximity is not lexically surfaced. The planned upgrade is embedding-distance scoring via `all-MiniLM-L6-v2` (22MB, fully local):

```python
# v0 (current)
overlap = len(axis_words & unit_words) / max(len(unit_words), 1)

# v1 (planned)
distance = 1 - cosine_similarity(embed(axis_text), embed(unit_text))
```

**Single-session scope.** No cross-session memory or axis recovery. For persistent memory across sessions, this pairs naturally with a system like [Hindsight](https://github.com/vectorize-io/hindsight): Hindsight manages the long-term store, the axis governs the per-turn window.

**Token estimation.** Count approximated as `char_length / 4`. Accurate enough for budget enforcement; replace with a tokenizer for exact counts.

---

## Paper

Full methodology, empirical results, and eviction curve data:

> Mohan, D.T. *Omega-Gated KV Eviction for Bounded-Memory LLM Sessions on Consumer Hardware.* Zenodo, March 2026. ORCID: 0000-0001-8054-2085. DOI: [add after deposit]

---

## Relation to Prior Work

The governance architecture (axis declaration, ratio-based coherence scoring, elastic context transition costs, append-only audit) derives from the Helical Imperative framework, documented progressively on Zenodo under ORCID 0000-0001-8054-2085 beginning December 2025. The STP (Solvency Transport Protocol) and IEP (Inclusive Epistemic Protocol) formalisms underlying this implementation appear in earlier deposits at that ORCID.

---

## License

CC0 1.0 Universal. No rights reserved.

This implementation is intended as a reproducible reference for the method. Fork it, extend it, cite it.

---

## Citation

```bibtex
@misc{mohan2026omega,
  author       = {Mohan, Deepak T.},
  title        = {Omega-Gated KV Eviction for Bounded-Memory LLM Sessions on Consumer Hardware},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {[https://doi.org/10.5281/zenodo.19207975]},
  url          = {https://github.com/dtmohan/omega-session},
  note         = {ORCID: 0000-0001-8054-2085}
}
```
