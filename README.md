# omega-session

**Ω-gated KV eviction and context governance for bounded-memory LLM sessions.**

A session governance method that maintains semantic coherence over arbitrarily long conversations within a fixed token budget. A structural coherence score (Ω) computed against a session axis declared at t=0 governs eviction of context entries — preserving semantically load-bearing content while displacing topically orthogonal noise.

Runs locally. No cloud. No external ML dependencies. Works with any Ollama model.

---

## Architecture Series

This repository is the reference implementation for two published technical notes:

| Note | Title | DOI |
|------|-------|-----|
| 1 | Omega-Gated KV Eviction for Bounded-Memory LLM Sessions on Consumer Hardware | [10.5281/zenodo.19207975](https://doi.org/10.5281/zenodo.19207975) |
| 2 | Recency Bias Is Architecture: Axis-First Ordering as a Structural Requirement for Context-Governed LLM Sessions | [10.5281/zenodo.19222620](https://doi.org/10.5281/zenodo.19222620) |

---

## Results

### Note 1 — Session governance (Qwen2.5-14B, Apple M3, 32GB, 20 turns)

| Metric | Governed | Baseline |
|--------|----------|----------|
| Throughput | **37.5 t/s** | 27.7 t/s |
| Speedup | **+35.3%** | — |
| Budget violations | **0 / 20 turns** | n/a |
| Early recall (planted fact) | **recalled** | lost |
| Final eviction rate | **57.3%** (51 of 89 units) | n/a |

### Note 2 — Recency bias (Qwen2.5-14B + Claude Sonnet 4, philosophical corpus)

| Condition | Budget 1100 | Budget 600 | Budget 450 |
|-----------|-------------|------------|------------|
| A — sliding window | 0/3 ✗ | 0/3 ✗ | 0/3 ✗ |
| B — lexical axis | 3/3 ✓ | 3/3 ✓ | **0/3 ✗** |
| C — depth-stratified | 3/3 ✓ | 3/3 ✓ [0,1] | 3/3 ✓ |
| D — perspective depth | 3/3 ✓ | 3/3 ✓ **[0,4]** | 3/3 ✓ |

Key findings:
- Recency bias is deterministic and model-agnostic — Sonnet and Qwen fail identically
- Conservative retrieval system prompts make recency bias **worse** (guided 0/3, blind 2/3)
- Axis-first ordering completely resolves Condition A at adequate budget — 0/3 → 3/3
- Lexical scorer degrades at stress 450 while structural position scorers hold
- Perspective depth pass (ego-dissolution) correctly promotes return units (POP) over descent units (PUSH) under budget pressure

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

**Axis-first placement invariant (Note 2):** the axis unit is always placed first in
the assembled context window — not merely retained through eviction, but positionally
first. This single rule deterministically resolves recency bias at adequate budget.

**STP structural minimum (Note 2):** size the context window from the text's own
argument structure, not by guessing:

```python
W = 500 + (stack_depth × 200) + (context_switches × 100)
```

Below W, structural information loss is guaranteed regardless of eviction policy or
model capability.

---

## Install

Requires Python 3.9+ and [Ollama](https://ollama.ai) running locally.

```bash
git clone https://github.com/dtmohan/omega-session
cd omega-session

# No pip install needed — stdlib only
# Pull your model if you haven't already
ollama pull qwen2.5:14b
```

For the Sonnet cross-model tests (Note 2): set `ANTHROPIC_API_KEY` in your environment.

---

## Usage

### Interactive session

```bash
python omega_session.py --model qwen2.5:14b --budget 3000 --verbose
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen2.5:14b` | Any Ollama model |
| `--budget` | `3000` | Max tokens fed per turn |
| `--verbose` | off | Show governance stats per turn |

Commands during session: `report` prints session stats, `quit` exits.

### T8 corpus tests (Note 2)

```bash
# Qwen — IEP-dynamic budget (auto-sized from argument structure)
python test_t8.py --model qwen2.5:14b

# Qwen — stress test below structural minimum
python test_t8.py --model qwen2.5:14b --budget 400

# Sonnet — 4-condition cross-model test (requires ANTHROPIC_API_KEY)
python test_t8_sonnet.py

# Perspective depth pass — ego-dissolution extension
python test_t8_perspective.py --model qwen2.5:14b
python test_t8_perspective.py --model qwen2.5:14b --stress 600
python test_t8_perspective.py --model qwen2.5:14b --stress 450
```

### Original test harness (Note 1)

```bash
# Unit tests only (no Ollama needed)
python test_harness.py --skip-live

# Full suite, 8 turns (~3 min)
python test_harness.py --model qwen2.5:14b --turns 8

# Stress run, 20 turns (~30 min)
python test_harness.py --model qwen2.5:14b --turns 20
```

---

## Files

```
omega_session.py         <- Entry point. CLI + Ollama client + session loop.
stp_session.py           <- Transport layer. Axis declaration, Ω scorer,
                            window builder, append-only audit log.
context_stack.py         <- Ingestion layer. Argument AST segmenter,
                            ContextUnit dataclass, STP structural minimum.
document_segmenter.py    <- Extended segmenter. Perspective depth pass,
                            anchor_density(), transition_friction(),
                            omega_direction(), perspective_eviction_priority().
test_harness.py          <- Note 1 test suite (T1-T6).
test_t8.py               <- Note 2 corpus test. IEP-dynamic + stress modes.
test_t8_sonnet.py        <- Note 2 cross-model test (Sonnet API, 4 conditions).
test_t8_perspective.py   <- Note 2 ego-dissolution / perspective depth pass.
```

No external dependencies except `anthropic` for `test_t8_sonnet.py`.
Everything imports upward from `context_stack.py`.

---

## Governance Invariants

Seven structural invariants enforced at every turn:

- **I1 Causality** — governance precedes the model call, always
- **I2 Conservation** — token budget is finite, tracked, enforced as hard ceiling
- **I3 Separation** — the scorer is architecturally separate from the generator
- **I4 Irreversibility** — the session audit log is append-only
- **I5 Identity** — the axis is declared at t=0, hash-committed, never updated
- **I6 Cost of Precision** — structural minimum sized from counters, not guessed
- **I7 Necessity of Validation** — FIN when underdetermined, never fabricate

---

## Known Limitations

**Lexical scorer.** The Ω scorer uses word-overlap against the axis text. This works
for natural conversation but fails for technically dense writing where semantic
proximity is not lexically surfaced. Note 2 shows this scorer degrades at stress 450
while structural position scorers hold. Planned upgrade: embedding-distance scoring.

**Perspective depth pass (regex-based).** The current ego-dissolution implementation
detects perspective entry/return via English discourse markers. The correct long-term
approach is grammar-based: spaCy dependency parse extracting ROOT subject changes and
subordinate clauses as PUSH/POP signals. Language-agnostic, no marker lists required.

**Single-session scope.** No cross-session axis persistence. The axis needs to be
re-established at each session open. For cross-session continuity, the committed
axis hash can be carried forward as an axis registry entry.

**Token estimation.** Count approximated as `char_length / 4`. Accurate enough for
budget enforcement; replace with a tokenizer for exact counts.

---

## Papers

> Mohan, D.T. *Omega-Gated KV Eviction for Bounded-Memory LLM Sessions on Consumer Hardware.* Helical Imperative — Architecture Series, Technical Note 1. Zenodo, March 2026. DOI: [10.5281/zenodo.19207975](https://doi.org/10.5281/zenodo.19207975)

> Mohan, D.T. *Recency Bias Is Architecture: Axis-First Ordering as a Structural Requirement for Context-Governed LLM Sessions.* Helical Imperative — Architecture Series, Technical Note 2. Zenodo, March 2026. DOI: [10.5281/zenodo.19222620](https://doi.org/10.5281/zenodo.19222620)

---

## Relation to Prior Work

The governance architecture derives from the Helical Imperative framework, documented
progressively on Zenodo under ORCID 0000-0001-8054-2085 beginning December 2025.
The STP kernel, IEP, and sSTP formalisms underlying this implementation appear in
earlier deposits at that ORCID. The perspective depth pass extends the Internal Ear
Protocol (IEP v1.0/v2.0, January–February 2026) to argument-level segmentation.

---

## License

CC0 1.0 Universal. No rights reserved.

Fork it, extend it, cite it.

---

## Citation

```bibtex
@misc{mohan2026omega,
  author       = {Mohan, Deepak T.},
  title        = {Omega-Gated KV Eviction for Bounded-Memory LLM Sessions on Consumer Hardware},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19207975},
  url          = {https://github.com/dtmohan/omega-session},
  note         = {ORCID: 0000-0001-8054-2085}
}

@misc{mohan2026recency,
  author       = {Mohan, Deepak T.},
  title        = {Recency Bias Is Architecture: Axis-First Ordering as a Structural Requirement for Context-Governed LLM Sessions},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19222620},
  url          = {https://github.com/dtmohan/omega-session},
  note         = {ORCID: 0000-0001-8054-2085}
}
```
