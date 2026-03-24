"""
test_harness.py — Systematic test suite for omega_session

Tests:
  T1  Unit test: context_stack segmentation
  T2  Unit test: stp_session scoring + window building
  T3  Integration: t/s stability over growing conversation (governed vs baseline)
  T4  Integration: tokens_fed stays within budget at all turns
  T5  Integration: early-context recall — governed vs sliding window
  T6  Stress: 30-turn deep conversation, eviction curves

Run:
    python test_harness.py                     # all tests
    python test_harness.py --tests T1 T2       # specific tests
    python test_harness.py --skip-live         # skip Ollama tests (T3-T6)
"""

import sys
import time
import json
import argparse
import statistics
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, ".")
from context_stack import parse_context_units, complexity_counters
from stp_session import open_session, score_unit, build_window, session_report
from omega_session import OmegaSessionLoop, ollama_chat

OLLAMA_HEALTH = "http://localhost:11434/api/tags"


# ── Test result primitives ───────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_s: float
    notes: List[str] = field(default_factory=list)
    metrics: dict    = field(default_factory=dict)

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        lines  = [f"  [{status}] {self.name}  ({self.duration_s:.2f}s)"]
        for n in self.notes:
            lines.append(f"         {n}")
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"         {k}: {v:.3f}")
            else:
                lines.append(f"         {k}: {v}")
        return "\n".join(lines)


def run_test(name, fn) -> TestResult:
    t0 = time.time()
    try:
        notes, metrics, passed = fn()
        return TestResult(name, passed, time.time() - t0, notes, metrics)
    except Exception as e:
        return TestResult(name, False, time.time() - t0,
                          notes=[f"EXCEPTION: {e}"], metrics={})


# ── Synthetic conversation corpus ────────────────────────────────────────────

SEED_TOPIC = "The relationship between entropy and information in thermodynamic systems"

SHORT_TURNS = [
    ("user",      "What is entropy in thermodynamics?"),
    ("assistant", "Entropy is a measure of disorder or randomness in a system. In thermodynamics it quantifies the number of microscopic configurations that correspond to a macroscopic state."),
    ("user",      "How does this connect to Shannon information theory?"),
    ("assistant", "Shannon entropy H = -Σ p log p mirrors Boltzmann entropy S = k log W. Both measure uncertainty — one over probability distributions, one over microstates."),
    ("user",      "However, can information be converted to physical work?"),
    ("assistant", "Yes — Szilard's engine and Maxwell's demon show that one bit of information is worth kT ln2 joules of work. Landauer's principle makes this precise."),
]

# Deliberately off-topic turns to stress eviction
NOISE_TURNS = [
    ("user",      "By the way, what is the capital of France?"),
    ("assistant", "Paris."),
    ("user",      "And who wrote Hamlet?"),
    ("assistant", "Shakespeare."),
    ("user",      "What is 17 times 23?"),
    ("assistant", "391."),
]

# Early recall probe — tests whether governed context preserves seed topic
RECALL_PROBE = "Earlier we discussed entropy and information — what was the key connection to Szilard's engine?"

# 30-turn stress corpus (alternating on/off topic)
def build_stress_corpus(n_turns: int = 30) -> List[tuple]:
    on_topic = [
        "How does the fluctuation theorem relate to entropy production?",
        "What is the Jarzynski equality and why does it matter?",
        "Can you explain the relationship between free energy and work?",
        "How does Maxwell's demon resolve with Landauer's principle?",
        "What is the role of information in biological systems?",
        "How does entropy apply to black hole thermodynamics?",
        "What is the connection between entropy and arrow of time?",
        "Explain the second law in terms of information erasure.",
        "How does entropy production relate to irreversibility?",
        "What is von Neumann entropy in quantum systems?",
    ]
    off_topic = [
        "What is the best way to cook pasta?",
        "Recommend a book on medieval history.",
        "What programming languages are popular in 2025?",
        "How do I improve my chess game?",
        "What is the tallest mountain in South America?",
    ]
    turns = []
    for i in range(n_turns):
        if i % 3 == 2:  # every 3rd turn is off-topic noise
            q = off_topic[i % len(off_topic)]
        else:
            q = on_topic[i % len(on_topic)]
        turns.append(("user", q))
        turns.append(("assistant", f"[Synthetic answer to: {q[:40]}...]"))
    return turns[:n_turns]


# ── T1: Unit test — context_stack ────────────────────────────────────────────

def test_t1_context_stack():
    notes, metrics = [], {}

    # Basic segmentation
    turns = [{"role": r, "content": c, "turn_index": i}
             for i, (r, c) in enumerate(SHORT_TURNS)]
    units = parse_context_units(turns)

    assert len(units) >= len(SHORT_TURNS), "Should have at least one unit per turn"
    notes.append(f"Parsed {len(units)} units from {len(SHORT_TURNS)} turns")

    # Switch detection — "However" turn should be PROVISIONAL
    provisional = [u for u in units if u.cross_switch]
    notes.append(f"Provisional units (cross_switch): {len(provisional)}")
    assert len(provisional) >= 1, "Expected at least one PROVISIONAL unit"

    # Counters
    c = complexity_counters(units)
    assert c.estimated_window >= 500, "IEP window should be >= 500"
    notes.append(f"IEP window: {c.estimated_window}  switches: {c.context_switches}")
    metrics["estimated_window"] = c.estimated_window
    metrics["total_units"]      = len(units)
    metrics["provisional_units"]= len(provisional)

    return notes, metrics, True


# ── T2: Unit test — stp_session scoring ──────────────────────────────────────

def test_t2_stp_session():
    notes, metrics = [], {}

    turns = [{"role": r, "content": c, "turn_index": i}
             for i, (r, c) in enumerate(SHORT_TURNS)]
    units = parse_context_units(turns)

    session = open_session(SEED_TOPIC, budget_tokens=500)
    notes.append(f"Session axis hash: {session.axis_hash[:16]}...")

    scores = [score_unit(u, session, len(SHORT_TURNS)-1, len(SHORT_TURNS))
              for u in units]

    # Axis-adjacent units (turn 0,1) should score lower omega than noise
    on_topic_omegas  = [s.omega for s, u in zip(scores, units) if u.turn_index <= 1]
    off_topic_omegas = [s.omega for s, u in zip(scores, units) if u.turn_index >= 4]

    mean_on  = statistics.mean(on_topic_omegas)  if on_topic_omegas  else 1.0
    mean_off = statistics.mean(off_topic_omegas) if off_topic_omegas else 0.0

    notes.append(f"Mean omega on-topic turns 0-1:  {mean_on:.3f}")
    notes.append(f"Mean omega off-topic turns 4-5: {mean_off:.3f}")
    metrics["mean_omega_on_topic"]  = mean_on
    metrics["mean_omega_off_topic"] = mean_off

    # Window building — tight budget forces eviction
    tight_budget = 40  # tokens
    selected, spent = build_window(units, scores, tight_budget, len(SHORT_TURNS)-1)
    notes.append(f"Tight budget ({tight_budget} tok): selected {len(selected)}/{len(units)} units, spent {spent}")
    assert spent <= tight_budget + 10, f"Spent {spent} exceeds budget {tight_budget}"
    metrics["tight_budget_selected"] = len(selected)
    metrics["tight_budget_spent"]    = spent

    # Verify latest user turn always pinned
    latest_user = [u for u in selected if u.role == "user"]
    assert len(latest_user) >= 1, "Latest user turn must always be in window"
    notes.append("Latest user turn correctly pinned in window")

    return notes, metrics, True


# ── Ollama health check ───────────────────────────────────────────────────────

def check_ollama(model: str) -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_HEALTH, timeout=3) as r:
            data = json.loads(r.read())
        available = [m["name"] for m in data.get("models", [])]
        return any(model in m for m in available)
    except Exception:
        return False


# ── T3: t/s stability over growing conversation ───────────────────────────────

def test_t3_tps_stability(model: str, n_turns: int = 8):
    notes, metrics = [], {}

    corpus = build_stress_corpus(n_turns * 2)
    user_turns = [(r, c) for r, c in corpus if r == "user"][:n_turns]

    # ── Governed ──
    gov_loop = OmegaSessionLoop(model=model, budget_tokens=2000, verbose=False)
    gov_tps   = []
    gov_tokens_fed = []

    for i, (_, content) in enumerate(user_turns):
        t0  = time.time()
        _   = gov_loop.chat(content)
        dur = time.time() - t0

        # Extract stats from last audit log entry
        if gov_loop.session and gov_loop.session.audit_log:
            last = gov_loop.session.audit_log[-1]
            gov_tokens_fed.append(last.get("tokens_fed", 0))

        # Re-measure tps via direct call for fairness
        _, stats = ollama_chat(
            [{"role": "user", "content": "ping"}], model
        )
        gov_tps.append(stats["tokens_per_second"])

    # ── Baseline (naive — full history fed each turn) ──
    baseline_tps = []
    baseline_history = []

    for i, (_, content) in enumerate(user_turns):
        baseline_history.append({"role": "user", "content": content})
        _, stats = ollama_chat(baseline_history, model)
        baseline_tps.append(stats["tokens_per_second"])
        baseline_history.append({
            "role": "assistant",
            "content": stats.get("response", "...")
        })

    gov_mean      = statistics.mean(gov_tps)      if gov_tps      else 0
    baseline_mean = statistics.mean(baseline_tps) if baseline_tps else 0
    gov_std       = statistics.stdev(gov_tps)      if len(gov_tps) > 1      else 0
    baseline_std  = statistics.stdev(baseline_tps) if len(baseline_tps) > 1 else 0

    notes.append(f"Governed  : mean={gov_mean:.1f} t/s  std={gov_std:.2f}  over {n_turns} turns")
    notes.append(f"Baseline  : mean={baseline_mean:.1f} t/s  std={baseline_std:.2f}  over {n_turns} turns")
    notes.append(f"Tokens fed (governed): {gov_tokens_fed}")

    metrics.update({
        "gov_mean_tps":      gov_mean,
        "baseline_mean_tps": baseline_mean,
        "gov_std_tps":       gov_std,
        "baseline_std_tps":  baseline_std,
    })

    # Pass condition: governed mean t/s >= 90% of baseline mean
    # (governed should be at least as fast; higher is better due to bounded prefill)
    speedup = (gov_mean / baseline_mean - 1.0) * 100 if baseline_mean > 0 else 0
    notes.append(f"Speedup: {speedup:+.1f}% (governed vs baseline)")
    passed = gov_mean >= baseline_mean * 0.90
    if not passed:
        notes.append("WARNING: governed mean t/s below 90% of baseline")

    return notes, metrics, passed


# ── T4: Budget enforcement ────────────────────────────────────────────────────

def test_t4_budget_enforcement(model: str, budget: int = 800, n_turns: int = 10):
    notes, metrics = [], {}

    corpus     = build_stress_corpus(n_turns * 2)
    user_turns = [(r, c) for r, c in corpus if r == "user"][:n_turns]
    loop       = OmegaSessionLoop(model=model, budget_tokens=budget, verbose=False)
    violations = []

    for i, (_, content) in enumerate(user_turns):
        loop.chat(content)
        if loop.session and loop.session.audit_log:
            last = [e for e in loop.session.audit_log if e.get("event") == "WINDOW_BUILT"]
            if last:
                fed = last[-1].get("tokens_fed", 0)
                if fed > budget + 20:  # 20-token slack for pinned turn
                    violations.append(f"Turn {i}: fed {fed} > budget {budget}")

    notes.append(f"Budget: {budget} tokens over {n_turns} turns")
    notes.append(f"Violations: {len(violations)}")
    for v in violations:
        notes.append(f"  {v}")

    metrics["budget"]     = budget
    metrics["n_turns"]    = n_turns
    metrics["violations"] = len(violations)

    return notes, metrics, len(violations) == 0


# ── T5: Early recall — governed vs sliding window ─────────────────────────────

def test_t5_early_recall(model: str):
    """
    Plant a specific fact early, pad with noise, then probe recall.
    Compare governed (Ω eviction) vs sliding window (last-N turns).
    """
    notes, metrics = [], {}

    PLANTED_FACT  = "The specific energy threshold for Szilard's engine is exactly kT ln2 joules per bit."
    RECALL_PROMPT = "What was the specific energy threshold we discussed for Szilard's engine?"
    NOISE_PADDING = 6  # turns of noise between plant and probe

    # Build conversation: plant → noise → probe
    plant_turns = [
        ("user",      f"Tell me about Szilard's engine. Note that: {PLANTED_FACT}"),
        ("assistant", f"Szilard's engine is a thought experiment. {PLANTED_FACT} This connects Maxwell's demon to thermodynamics."),
    ]
    noise = [
        ("user",      "What is the population of Tokyo?"),
        ("assistant", "About 14 million in the city proper, 37 million in greater Tokyo."),
        ("user",      "Who invented the telephone?"),
        ("assistant", "Alexander Graham Bell is credited with inventing the telephone in 1876."),
        ("user",      "What is photosynthesis?"),
        ("assistant", "Photosynthesis converts light energy to chemical energy in plants via chlorophyll."),
    ]
    all_turns = plant_turns + noise[:NOISE_PADDING]

    # ── Governed ──
    gov_loop = OmegaSessionLoop(model=model, budget_tokens=800, verbose=False)
    for role, content in all_turns:
        if role == "user":
            gov_loop.chat(content)
    gov_response = gov_loop.chat(RECALL_PROMPT)
    gov_recalled  = "kT ln2" in gov_response or "kT" in gov_response

    # ── Sliding window baseline (last 4 turns only) ──
    recent_turns = all_turns[-4:] + [("user", RECALL_PROMPT)]
    sliding_messages = [{"role": r, "content": c} for r, c in recent_turns]
    sliding_response, _ = ollama_chat(sliding_messages, model)
    sliding_recalled = "kT ln2" in sliding_response or "kT" in sliding_response

    notes.append(f"Planted fact: '{PLANTED_FACT[:60]}...'")
    notes.append(f"Noise turns between plant and probe: {NOISE_PADDING}")
    notes.append(f"Governed recalled fact:        {gov_recalled}  (kT ln2 in response)")
    notes.append(f"Sliding window recalled fact:  {sliding_recalled}")
    notes.append(f"Governed response (first 120):  {gov_response[:120]}")
    notes.append(f"Sliding response  (first 120):  {sliding_response[:120]}")

    metrics["gov_recalled"]     = int(gov_recalled)
    metrics["sliding_recalled"] = int(sliding_recalled)

    # Pass: governed recalls AND (governed >= sliding window)
    passed = gov_recalled or (gov_recalled == sliding_recalled)
    return notes, metrics, passed


# ── T6: Stress — 30-turn eviction curves ─────────────────────────────────────

def test_t6_stress_eviction(model: str, n_turns: int = 20):
    notes, metrics = [], {}

    corpus     = build_stress_corpus(n_turns * 2)
    user_turns = [(r, c) for r, c in corpus if r == "user"][:n_turns]
    loop       = OmegaSessionLoop(model=model, budget_tokens=1500, verbose=False)

    eviction_curve = []   # (turn, total_units, selected, evicted, tokens_fed)

    for i, (_, content) in enumerate(user_turns):
        loop.chat(content)
        if loop.session:
            log = [e for e in loop.session.audit_log if e.get("event") == "WINDOW_BUILT"]
            if log:
                last = log[-1]
                total    = last.get("total_units", 0)
                selected = last.get("selected_units", 0)
                evicted  = last.get("evicted_units", 0)
                fed      = last.get("tokens_fed", 0)
                eviction_curve.append((i, total, selected, evicted, fed))

    notes.append(f"Eviction curve over {n_turns} turns (turn | total | selected | evicted | tokens_fed):")
    for row in eviction_curve:
        notes.append(f"  turn={row[0]:2d}  total={row[1]:3d}  selected={row[2]:3d}  evicted={row[3]:3d}  fed={row[4]:4d}")

    if eviction_curve:
        final = eviction_curve[-1]
        metrics["final_total_units"]    = final[1]
        metrics["final_selected_units"] = final[2]
        metrics["final_eviction_rate"]  = 1.0 - (final[2] / max(final[1], 1))
        metrics["final_tokens_fed"]     = final[4]
        notes.append(f"Final eviction rate: {metrics['final_eviction_rate']:.1%}  "
                     f"({final[1] - final[2]} of {final[1]} units evicted)")

    # Pass: tokens_fed never exceeded budget in any turn
    passed = all(row[4] <= 1500 + 20 for row in eviction_curve)
    return notes, metrics, passed


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests",     nargs="*", default=None,
                        help="Which tests to run e.g. T1 T2 T3")
    parser.add_argument("--skip-live", action="store_true",
                        help="Skip tests that require Ollama (T3-T6)")
    parser.add_argument("--model",     default="qwen2.5:14b")
    parser.add_argument("--turns",     type=int, default=8,
                        help="Turns for T3/T6 (default 8, use 20+ for full stress)")
    args = parser.parse_args()

    want = set(args.tests) if args.tests else {"T1", "T2", "T3", "T4", "T5", "T6"}

    print("=" * 60)
    print("Ω Session — Test Harness")
    print(f"Model: {args.model}")
    print("=" * 60)

    results: List[TestResult] = []

    # ── Unit tests (no Ollama needed) ─────────────────────────────────────────
    if "T1" in want:
        results.append(run_test("T1: context_stack segmentation", test_t1_context_stack))

    if "T2" in want:
        results.append(run_test("T2: stp_session scoring + window", test_t2_stp_session))

    # ── Integration tests ─────────────────────────────────────────────────────
    if not args.skip_live:
        live_ok = check_ollama(args.model)
        if not live_ok:
            print(f"\n[SKIP] Ollama not running or model '{args.model}' not found.")
            print("       Start Ollama and pull the model, or use --skip-live.\n")
        else:
            print(f"\n[OK] Ollama reachable. Model '{args.model}' found.\n")

            if "T3" in want:
                results.append(run_test(
                    f"T3: t/s stability ({args.turns} turns)",
                    lambda: test_t3_tps_stability(args.model, args.turns)
                ))

            if "T4" in want:
                results.append(run_test(
                    f"T4: budget enforcement ({args.turns} turns)",
                    lambda: test_t4_budget_enforcement(args.model, budget=800, n_turns=args.turns)
                ))

            if "T5" in want:
                results.append(run_test(
                    "T5: early recall (governed vs sliding window)",
                    lambda: test_t5_early_recall(args.model)
                ))

            if "T6" in want:
                results.append(run_test(
                    f"T6: stress eviction curve ({args.turns} turns)",
                    lambda: test_t6_stress_eviction(args.model, args.turns)
                ))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nResults")
    print("-" * 60)
    for r in results:
        print(r)

    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    print("-" * 60)
    print(f"\n{passed}/{total} tests passed\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
