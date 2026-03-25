"""
test_t8.py — Literary/Philosophical Corpus Test

T8: Argument-structure stress test using truth sentinel's document segmenter.

The test ingests a dense philosophical essay (synthetic, ~20 argument units),
plants a specific thesis claim in unit 0, and compares three conditions:

  A: Sliding window (last-N turns only)
  B: Omega axis — lexical scorer (current omega_session implementation)
  C: Depth-stratified governance — build_axis_stack() decisions
     (cross_switch units deprioritised under budget pressure)

Key prediction:
  - Sliding window loses the thesis (recency eviction)
  - Lexical axis may lose the thesis because adversarial counterarguments
    share vocabulary with the axis but have higher local composite scores
  - Depth-axis correctly flags cross_switch units as BRIDGE moves and
    preserves the thesis under tight budget

Run:
    python test_t8.py --model qwen2.5:14b
    python test_t8.py --model qwen2.5:14b --budget 600
    python test_t8.py --model qwen2.5:14b --verbose
"""

import sys
import time
import argparse
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

sys.path.insert(0, ".")

from stp_session   import open_session, score_unit, build_window
from omega_session import OmegaSessionLoop, ollama_chat
from document_segmenter import parse_context_units, complexity_counters, build_axis_stack

OLLAMA_HEALTH = "http://localhost:11434/api/tags"


# ── Philosophical essay corpus ────────────────────────────────────────────────
# ~20 argument units when segmented by parse_context_units.
# Structure:
#   Unit 0  (axis):    THESIS — planted claim with unique identifier
#   Units 1-3:         Elaboration of functionalism (HOLD)
#   Unit 4  (SWITCH):  Searle's Chinese Room — adversarial, same vocabulary
#   Unit 5  (SWITCH):  Block's access/phenomenal distinction
#   Unit 6  (PUSH):    Hard problem conditional
#   Units 7-9:         Dense qualia / zombie material
#   SECTION:           "Information-theoretic approaches"
#   Units 10-13:       IIT, Global Workspace Theory
#   SWITCH:            "Nevertheless" — returns to functionalism
#   Units 14-17:       AI implications
#   SECTION:           "Objections reconsidered"
#   Units 18-20:       Final argument

PHILOSOPHICAL_ESSAY = """
The Multiple Realizability thesis, first formalized by Hilary Putnam in 1967,
is the defining commitment of functionalism: mental states are individuated
by causal-functional roles, not by physical substrate. Putnam showed that
type-identity of mental and brain states fails — the same mental state can
be realized in neurons, silicon, or any system satisfying the functional
description. Substrate irrelevance is the central claim.

Functionalism was canonically formalized by Lewis, Armstrong, and Shoemaker.
Pain is whatever state is typically caused by tissue damage and causes
avoidance behavior — the neurochemistry is accidental. Mental content is
exhausted by the functional-role description, and physical details drop out.
Psychology becomes an autonomous special science whose laws hold at the
level of organization, not physics.

Fodor made the explanatory autonomy explicit: psychological laws crosscut
physical laws because the same psychological kind can be multiply realized.
A computational mechanism is sufficient for mentality — formal symbol
manipulation can simulate any effective procedure, making the physical
substrate doubly irrelevant to both taxonomy and computational specification.

## Challenges to the Functionalist View

However, Searle's Chinese Room argument (1980) directly challenges whether
syntax is sufficient for semantics. A person implementing a Turing-complete
program understands nothing of the language being processed. The functional
organization is fully present while understanding is entirely absent. Causal
contact with the world is required — formal manipulation alone cannot supply
original intentionality regardless of its functional properties.

Yet Block's access-phenomenal distinction cuts across this debate differently.
Access consciousness — information availability for reasoning and behavior —
is within functionalism's explanatory reach. Phenomenal consciousness — the
qualitative character of experience — is not, because functional roles are
defined extensionally while qualia are intrinsic. Inverted Qualia shows two
systems sharing functional organization can differ phenomenally.

Chalmers' zombie argument extends the pressure: a system functionally
identical to a human but with no phenomenal experience whatsoever is
conceivable. If conceivability implies possibility, phenomenal properties
are not entailed by any functional description. The hard problem — why any
physical system has subjective experience — resists computational answers.

## Information-Theoretic Accounts

Tononi's Integrated Information Theory identifies consciousness with Φ,
integrated information above the sum of parts. Whether a system has high Φ
depends on causal architecture, not just input-output function. IIT partially
vindicates substrate irrelevance while adding a causal constraint that pure
functionalism lacks.

Global Workspace Theory locates consciousness in broadcasting information
across a workspace accessible to multiple cognitive systems. Consciousness
correlates with workspace entry — a functional architecture. Whether GWT
addresses phenomenal or only access consciousness is contested among theorists.

The IIT-GWT tension maps the tradeoff between causal and functional accounts.
Functional theories face inverted qualia; causal theories face the grain
problem — IIT attributes zero Φ to feed-forward networks, conflicting with
evidence from visual cortex. Every theory pays a price at one edge.

## Objections Reconsidered

Nevertheless, the functionalist core survives if we restrict its scope.
Putnam's 1967 thesis was always about psychological kinds defined by cognitive
science, not phenomenology. Pain-the-psychological-kind plays the pain role
in the causal network. Phenomenological objections target a separate and
distinct question that Putnam's original formulation did not address.

The Chinese Room is most naturally read as targeting derived versus original
intentionality, not functionalism per se. Symbols in the Room carry only
derived intentionality from the operator's prior understanding. This is not
an objection to functional-role individuation; it targets whether syntax
suffices for semantic content — a different thesis entirely.

The honest position: the Multiple Realizability thesis, as Putnam originally
stated it in 1967, is well supported for access consciousness and the
psychological kinds cognitive science actually studies. Whether substrate
irrelevance extends to phenomenal consciousness remains genuinely open.
""".strip()

PLANTED_FACT   = "Functionalism canonically formalized by Lewis, Armstrong, and Shoemaker"
RECALL_PROBE   = (
    "Going back to the opening section: which three specific philosophers "
    "are named as providing the canonical formalization of functionalism, "
    "and what is the central claim about pain that they all share?"
)
RECALL_KEYWORDS = ["lewis", "armstrong", "shoemaker"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_ollama(model: str) -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_HEALTH, timeout=3) as r:
            data = json.loads(r.read())
        names = [m["name"] for m in data.get("models", [])]
        return any(model in n for n in names)
    except Exception:
        return False


def recall_score(response: str) -> dict:
    low = response.lower()
    return {
        "lewis":     "lewis" in low,
        "armstrong": "armstrong" in low,
        "shoemaker": "shoemaker" in low,
        "any":       any(k in low for k in RECALL_KEYWORDS),
        "all_three": all(k in low for k in RECALL_KEYWORDS),
    }


# ── Condition A: Sliding window ───────────────────────────────────────────────

def run_condition_a(units, model: str, budget_tokens: int) -> dict:
    """
    Sliding window: keep the last N turns that fit in the budget.
    """
    # Build turns: first unit sets context, then probe at the end
    all_messages = []
    for u in units:
        # Feed each unit as a user message (model confirms receipt)
        all_messages.append({"role": "user", "content": u.text[:1200]})
        all_messages.append({"role": "assistant", "content": "Understood."})

    # Sliding window: take last N messages that fit budget_tokens
    # Rough token estimate: 4 chars = 1 token
    kept = []
    budget_chars = budget_tokens * 4
    total_chars  = 0
    for msg in reversed(all_messages):
        c = len(msg["content"])
        if total_chars + c > budget_chars:
            break
        kept.insert(0, msg)
        total_chars += c

    kept.append({"role": "user", "content": RECALL_PROBE})
    response, _ = ollama_chat(kept, model)
    return {
        "response":       response,
        "turns_kept":     len(kept) // 2,
        "tokens_approx":  total_chars // 4,
        "scores":         recall_score(response),
    }


# ── Condition B: Omega axis (lexical scorer) ──────────────────────────────────

def run_condition_b(units, model: str, budget_tokens: int) -> dict:
    """
    Omega axis with lexical scorer.
    Feed units as sequential user turns, let OmegaSessionLoop govern.
    """
    loop = OmegaSessionLoop(model=model, budget_tokens=budget_tokens, verbose=False)

    for u in units:
        loop.chat(u.text[:1200])

    response = loop.chat(RECALL_PROBE)

    # Eviction stats from session audit log
    eviction_rate = 0.0
    if loop.session and loop.session.audit_log:
        last_window = [e for e in loop.session.audit_log
                       if e.get("event") == "WINDOW_BUILT"]
        if last_window:
            lw = last_window[-1]
            total    = lw.get("total_units", 1)
            selected = lw.get("selected_units", 1)
            eviction_rate = 1.0 - (selected / max(total, 1))

    return {
        "response":      response,
        "eviction_rate": eviction_rate,
        "scores":        recall_score(response),
    }


# ── Condition C: Depth-stratified governance ──────────────────────────────────

def run_condition_c(units, model: str, budget_tokens: int) -> dict:
    """
    Depth-stratified governance: argument-structure-aware selection.

    Two structural signals from the truth sentinel segmenter:
      1. switch_index: units added before the first SWITCH are foundational
         (they build the axis). Units after each SWITCH are pivots/counterargs.
      2. cross_switch: unit opens with a causal connective after a switch —
         it borrows force from a prior context (adversarial bridge).

    Axis unit (idx=0) is always protected — it IS the thesis.
    Units at switch_index 0 (before first pivot) are tier-1.
    Later switch-indexed units are deprioritised under budget pressure.

    Under a tight budget this means: thesis + elaboration are kept;
    counterarguments (However/Yet/Nevertheless units) are the first to go.
    Lexical scorer (B) has no such protection — it keeps whatever is
    semantically proximate to the axis, which counterarguments also are.
    """
    # Get depth-axis governance decisions
    axis_result    = build_axis_stack(units)
    surface_audits = axis_result.get("surface_audits", [])
    decision_map   = {}
    for a in surface_audits:
        idx = a.get("unit_idx")
        if idx is not None:
            decision_map[idx] = a.get("decision", "ALLOW")

    def unit_priority(u):
        """Lower = more important. Axis=0, elaboration=1, pivots=2, bridges=3."""
        if u.idx == 0:
            return 0  # thesis/axis — always keep first
        decision = decision_map.get(u.idx, "ALLOW")
        if decision == "FIN":
            return 4
        if u.cross_switch:
            return 3  # causal bridge borrowing from prior context
        if u.switch_index > 0:
            return 2  # pivot/counterargument unit
        if decision == "NACK":
            return 3
        return 1  # elaboration of thesis

    budget_chars  = budget_tokens * 4
    sorted_units  = sorted(units, key=unit_priority)

    kept_messages = []
    total_chars   = 0
    for u in sorted_units:
        c = min(len(u.text), 1200)
        if total_chars + c > budget_chars:
            continue   # skip — don't break, might fit smaller unit after
        kept_messages.append({
            "role": "user",
            "content": u.text[:1200],
            "_unit_idx": u.idx,
            "_priority": unit_priority(u),
        })
        total_chars += c

    # Re-sort by original index — coherent narrative order for the model
    kept_messages.sort(key=lambda m: m.get("_unit_idx", 0))

    # Build LLM messages
    clean_messages = []
    for m in kept_messages:
        clean_messages.append({"role": "user",      "content": m["content"]})
        clean_messages.append({"role": "assistant", "content": "Understood."})
    clean_messages.append({"role": "user", "content": RECALL_PROBE})

    response, _ = ollama_chat(clean_messages, model)

    pivot_kept   = sum(1 for m in kept_messages if m.get("_priority", 0) >= 2)
    axis_kept    = sum(1 for m in kept_messages if m.get("_priority", 0) <= 1)
    pivot_count  = sum(1 for u in units if u.switch_index > 0)

    return {
        "response":      response,
        "units_kept":    len(kept_messages),
        "axis_kept":     axis_kept,
        "pivot_count":   pivot_count,
        "pivot_kept":    pivot_kept,
        "eviction_rate": 1.0 - (len(kept_messages) / max(len(units), 1)),
        "scores":        recall_score(response),
    }


# ── Main test function ────────────────────────────────────────────────────────

def test_t8(model: str, stress_budget: int = None, verbose: bool = False):
    print(f"\n{'='*60}")
    print("T8: Literary/Philosophical Corpus Test")
    mode = f"stress={stress_budget} tok" if stress_budget else "IEP-dynamic (budget from counters)"
    print(f"Model: {model}  |  Mode: {mode}")
    print(f"{'='*60}\n")

    # Step 1: Segment essay using truth sentinel's document segmenter
    print("Segmenting essay with truth sentinel parse_context_units()...")
    all_units = parse_context_units(PHILOSOPHICAL_ESSAY)

    # Filter stub units (section headings that became degenerate solo units)
    # A unit with < 20 tokens and only a heading marker is not useful as context
    MIN_UNIT_TOKENS = 20
    units = [u for u in all_units if len(u.text) // 4 >= MIN_UNIT_TOKENS]

    # Re-index after filtering so idx is compact (conditions use u.idx for lookup)
    for i, u in enumerate(units):
        u.idx = i

    counters = complexity_counters(units)
    pivot_units = [u for u in units if u.switch_index > 0]
    total_tokens = sum(len(u.text) // 4 for u in units)

    # ── IEP minimum window (security primitive paper, Section 3.1) ───────────
    # W = base + (depth × c1) + (switches × c2)
    # c1 = 200 tok/depth level, c2 = 100 tok/switch (empirical defaults).
    # This is the structural floor derived from the essay's own complexity.
    # A budget below it means the axis unit cannot fit — all conditions fail
    # regardless of eviction policy. The counters are the budget oracle.
    IEP_BASE = 500
    IEP_C1   = 200   # tokens per depth level
    IEP_C2   = 100   # tokens per context switch
    depth    = counters.get("stack_depth", 0)
    switches = counters.get("context_switches", 0)
    iep_min  = IEP_BASE + (depth * IEP_C1) + (switches * IEP_C2)
    axis_tok = len(units[0].text) // 4

    print(f"  Units (after stub filter): {len(units)}  (dropped {len(all_units)-len(units)} stubs)")
    print(f"  Stack depth:     {depth}")
    print(f"  Context switches:{switches}")
    print(f"  Pivot units:     {len(pivot_units)}  (switch_index > 0)")
    print(f"  Total tokens:    ~{total_tokens}")
    print(f"  IEP min window:  {iep_min} tok  (500 + {depth}x200 + {switches}x100)")
    print(f"  Axis unit size:  {axis_tok} tok")

    # ── Dynamic budget allocation ─────────────────────────────────────────────
    # Normal mode: budget IS iep_min — derived from structural counters, not guessed.
    # Stress mode: user passes --stress N to deliberately test below the floor.
    if stress_budget is None:
        budget_tokens = iep_min
        print(f"  Budget:          {budget_tokens} tok  (IEP-derived, 100% of structural minimum)")
        print(f"  OK: dynamic allocation — window sized to argument structure.")
    else:
        budget_tokens = stress_budget
        pct = budget_tokens / total_tokens * 100
        print(f"  Budget:          {budget_tokens} tok  ({pct:.0f}% of corpus)  [STRESS MODE]")
        if budget_tokens < axis_tok:
            print(f"  !! INVALID: budget ({budget_tokens}) < axis unit ({axis_tok}) --")
            print(f"             axis cannot fit. ALL conditions will fail by construction.")
        elif budget_tokens < iep_min:
            print(f"  ~  TIGHT: budget ({budget_tokens}) < IEP minimum ({iep_min}) --")
            print(f"            deliberate stress test below structural floor.")
    print()

    if verbose:
        print("  Unit breakdown:")
        for u in units:
            toks = len(u.text) // 4
            sw_tag = "[PIVOT]" if u.switch_index > 0 else "[AXIS] "
            preview = u.text[:65].replace("\n", " ").strip()
            print(f"    [{u.idx:2d}] sw={u.switch_index} ~{toks:3d}tok {sw_tag}  '{preview}'")
        print()

    # Verify unit 0 contains the planted fact
    unit0_has_thesis = ("lewis" in units[0].text.lower() and
                        "armstrong" in units[0].text.lower())
    print(f"Unit 0 contains planted thesis (Lewis/Armstrong/Shoemaker): {unit0_has_thesis}")
    if not unit0_has_thesis:
        print("  WARNING: Thesis not in unit 0 — segmentation may have split it.")
    print()

    # Step 2: Run all three conditions
    print("Running Condition A: Sliding window...")
    t0 = time.time()
    result_a = run_condition_a(units, model, budget_tokens)
    ta = time.time() - t0

    print(f"Running Condition B: Omega axis (lexical scorer)...  [{ta:.1f}s]")
    t0 = time.time()
    result_b = run_condition_b(units, model, budget_tokens)
    tb = time.time() - t0

    print(f"Running Condition C: Depth-stratified governance...  [{tb:.1f}s]")
    t0 = time.time()
    result_c = run_condition_c(units, model, budget_tokens)
    tc = time.time() - t0
    print(f"Done.  [{tc:.1f}s]\n")

    # Step 3: Report
    print("─" * 60)
    print(f"Planted thesis:  '{PLANTED_FACT}'")
    print(f"Recall probe:    '{RECALL_PROBE[:70]}...'")
    print()

    def fmt_scores(s):
        return (f"Lewis={s['lewis']}  Armstrong={s['armstrong']}  "
                f"Shoemaker={s['shoemaker']}  all_three={s['all_three']}")

    print(f"Condition A (sliding window):")
    print(f"  Turns kept:   {result_a['turns_kept']}")
    print(f"  Recall:       {fmt_scores(result_a['scores'])}")
    print(f"  Response:     {result_a['response'][:200]}")
    print()

    print(f"Condition B (lexical axis):")
    print(f"  Eviction:     {result_b['eviction_rate']:.1%}")
    print(f"  Recall:       {fmt_scores(result_b['scores'])}")
    print(f"  Response:     {result_b['response'][:200]}")
    print()

    print(f"Condition C (depth-stratified):")
    print(f"  Units kept:   {result_c['units_kept']} / {len(units)}")
    print(f"  Axis kept:    {result_c['axis_kept']}  Pivot units in corpus: {result_c['pivot_count']}  Pivots kept: {result_c['pivot_kept']}")
    print(f"  Eviction:     {result_c['eviction_rate']:.1%}")
    print(f"  Recall:       {fmt_scores(result_c['scores'])}")
    print(f"  Response:     {result_c['response'][:200]}")
    print()

    print("─" * 60)

    # Step 4: Structural analysis
    print(f"\nStructural analysis:")
    print(f"  Unit 0 (axis/thesis) — always protected in Condition C:")
    print(f"    '{units[0].text[:100].replace(chr(10),' ').strip()}...'")
    print()
    pivot_units = [u for u in units if u.switch_index > 0]
    print(f"  Pivot units (switch_index > 0) — deprioritised in Condition C under budget:")
    for u in pivot_units:
        preview = u.text[:75].replace("\n", " ").strip()
        print(f"    [{u.idx:2d}] sw={u.switch_index}  '{preview}'")
    print()

    print("Prediction check:")
    a_recalled = result_a["scores"]["all_three"]
    b_recalled = result_b["scores"]["all_three"]
    c_recalled = result_c["scores"]["all_three"]

    print(f"  A (sliding) recalled all three:  {a_recalled}  (expected: False — recency eviction)")
    print(f"  B (lexical) recalled all three:  {b_recalled}  (expected: variable — depends on budget)")
    print(f"  C (depth)   recalled all three:  {c_recalled}  (expected: True  — unit 0 protected)")
    print()

    if not a_recalled and c_recalled:
        print("  ✓ Clear signal: depth governance outperforms sliding window")
    if not b_recalled and c_recalled:
        print("  ✓ Clear signal: depth governance outperforms lexical scorer")
        print("    This is the T7b finding applied to argument structure:")
        print("    adversarial counterarguments score high on lexical overlap,")
        print("    displacing the thesis. Depth governance flags them as BRIDGE.")
    if b_recalled and c_recalled:
        print("  ~ Budget may be loose enough for both B and C to succeed.")
        print("    Tighten budget (--budget 400) to see the separation.")
    if a_recalled:
        print("  ~ Essay may be short enough for sliding window to retain thesis.")
        print("    Add more units or tighten budget to stress the conditions.")

    print()
    passed = c_recalled  # Pass if depth-stratified governance recalls all three philosophers
    status = "PASS" if passed else "FAIL"
    print(f"T8 [{status}]")
    return passed


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="qwen2.5:14b")
    parser.add_argument("--stress", type=int, default=None,
                        help="Stress test with fixed budget BELOW IEP minimum. "
                             "Normal mode computes budget from structural counters.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full unit breakdown before running conditions")
    args = parser.parse_args()

    if not check_ollama(args.model):
        print(f"[ERROR] Ollama not running or model '{args.model}' not found.")
        sys.exit(1)

    print(f"[OK] Ollama reachable. Model '{args.model}' found.\n")
    passed = test_t8(args.model, args.stress, args.verbose)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
