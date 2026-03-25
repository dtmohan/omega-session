"""
test_t8_perspective.py — T8 with ego-dissolution perspective depth pass

Runs the philosophical corpus through four conditions:
  A) Sliding window
  B) Lexical axis (omega_session)
  C) Depth-stratified (switch_index — current Condition C)
  D) Perspective depth (perspective_eviction_priority — ego-dissolution)

The key question: does perspective-aware eviction improve on switch_index
alone? Specifically for the "Nevertheless..." unit — switch_index ranks it
as a low-priority pivot (sw=6). Perspective depth ranks it as a POP
(returning to axis, priority tier 2) — it should be KEPT, not evicted,
because it is axis-convergent.

Run standalone:
    python test_t8_perspective.py --model qwen2.5:14b

Or integrate into test_harness_v2.py:
    from test_t8_perspective import test_t8_perspective
    # register as T8p
"""

import sys
import time
import argparse

sys.path.insert(0, ".")

from test_t8 import (
    PHILOSOPHICAL_ESSAY, PLANTED_FACT, RECALL_PROBE, RECALL_KEYWORDS, recall_score
)
from document_segmenter import (
    parse_context_units, complexity_counters,
    perspective_depth_pass, perspective_eviction_priority, anchor_density
)
from omega_session import OmegaSessionLoop, ollama_chat

OLLAMA_HEALTH = "http://localhost:11434/api/tags"


def check_ollama(model: str) -> bool:
    import json, urllib.request, urllib.error
    try:
        with urllib.request.urlopen(OLLAMA_HEALTH, timeout=3) as r:
            data = json.loads(r.read())
        return any(model in m["name"] for m in data.get("models", []))
    except Exception:
        return False


def test_t8_perspective(model: str, stress_budget: int = None):
    """
    T8 with perspective depth pass.
    Returns (notes, metrics, passed) for test_harness_v2.py run_test().
    """
    notes, metrics = [], {}

    # Segment
    all_units = parse_context_units(PHILOSOPHICAL_ESSAY)
    units     = [u for u in all_units if len(u.text) // 4 >= 20]
    for i, u in enumerate(units):
        u.idx = i

    counters = complexity_counters(units)
    depth    = counters.get("stack_depth", 0)
    switches = counters.get("context_switches", 0)
    stp_min  = 500 + depth * 200 + switches * 100
    budget   = stress_budget if stress_budget else stp_min

    # Run perspective pass
    p_units = perspective_depth_pass(units)

    notes.append(f"Essay: {len(units)} units  stp_min={stp_min}  budget={budget}")
    notes.append(f"Perspective analysis:")
    for pu in p_units:
        prio = perspective_eviction_priority(pu)
        notes.append(
            f"  [{pu.idx}] op={pu.perspective_op:5s} depth={pu.perspective_depth} "
            f"ad={pu.anchor_density:.2f} omega_dir={pu.omega_direction:+.1f} "
            f"priority={prio}  '{pu.text[:55].replace(chr(10),' ').strip()}'"
        )
    notes.append("")

    # ── Condition A: Sliding window ───────────────────────────────────────────
    all_messages = []
    for u in units:
        all_messages.append({"role": "user",      "content": u.text[:1200]})
        all_messages.append({"role": "assistant", "content": "Understood."})

    kept_a, total_chars = [], 0
    for msg in reversed(all_messages):
        c = len(msg["content"])
        if total_chars + c > budget * 4:
            break
        kept_a.insert(0, msg)
        total_chars += c
    kept_a.append({"role": "user", "content": RECALL_PROBE})

    try:
        if len(kept_a) <= 1:
            # Only the probe — no context kept at all
            resp_a = "[NO CONTEXT IN WINDOW — budget too small for any unit]"
        else:
            resp_a, _ = ollama_chat(kept_a, model)
        score_a = recall_score(resp_a)
        notes.append(f"A (sliding):       {score_a['all_three']}  "
                     f"L={score_a['lewis']} A={score_a['armstrong']} "
                     f"S={score_a['shoemaker']}  kept={len(kept_a)-1} turns")
    except Exception as e:
        score_a = recall_score("")
        notes.append(f"A (sliding):       TIMEOUT/ERROR — {str(e)[:60]}")

    # ── Condition B: Lexical axis ─────────────────────────────────────────────
    loop_b = OmegaSessionLoop(model=model, budget_tokens=budget, verbose=False)
    for u in units:
        loop_b.chat(u.text[:1200])
    resp_b  = loop_b.chat(RECALL_PROBE)
    score_b = recall_score(resp_b)
    notes.append(f"B (lexical):       {score_b['all_three']}  "
                 f"L={score_b['lewis']} A={score_b['armstrong']} S={score_b['shoemaker']}")

    # ── Condition C: Switch-index (current) ───────────────────────────────────
    def switch_priority(u):
        if u.idx == 0: return 0
        if u.switch_index > 0: return 2
        return 1

    budget_chars = budget * 4
    c_sorted  = sorted(units, key=switch_priority)
    c_kept, c_chars = [], 0
    for u in c_sorted:
        c = min(len(u.text), 1200)
        if c_chars + c > budget_chars:
            continue
        c_kept.append(u)
        c_chars += c
    c_kept.sort(key=lambda u: u.idx)

    clean_c = []
    for u in c_kept:
        clean_c.append({"role": "user",      "content": u.text[:1200]})
        clean_c.append({"role": "assistant", "content": "Understood."})
    clean_c.append({"role": "user", "content": RECALL_PROBE})
    resp_c, _ = ollama_chat(clean_c, model)
    score_c   = recall_score(resp_c)
    c_ids = [u.idx for u in c_kept]
    notes.append(f"C (switch_index):  {score_c['all_three']}  "
                 f"L={score_c['lewis']} A={score_c['armstrong']} S={score_c['shoemaker']}  "
                 f"kept={c_ids}")

    # ── Condition D: Perspective depth (ego-dissolution) ──────────────────────
    d_sorted  = sorted(p_units, key=perspective_eviction_priority)
    d_kept, d_chars = [], 0
    for pu in d_sorted:
        c = min(len(pu.text), 1200)
        if d_chars + c > budget_chars:
            continue
        d_kept.append(pu)
        d_chars += c
    d_kept.sort(key=lambda pu: pu.idx)

    clean_d = []
    for pu in d_kept:
        clean_d.append({"role": "user",      "content": pu.text[:1200]})
        clean_d.append({"role": "assistant", "content": "Understood."})
    clean_d.append({"role": "user", "content": RECALL_PROBE})
    resp_d, _ = ollama_chat(clean_d, model)
    score_d   = recall_score(resp_d)
    d_ids = [pu.idx for pu in d_kept]
    notes.append(f"D (perspective):   {score_d['all_three']}  "
                 f"L={score_d['lewis']} A={score_d['armstrong']} S={score_d['shoemaker']}  "
                 f"kept={d_ids}")

    # ── Key comparison: what D kept vs C kept ────────────────────────────────
    notes.append("")
    notes.append("Selection comparison (C vs D):")
    for pu in p_units:
        in_c = pu.idx in c_ids
        in_d = pu.idx in d_ids
        prio_c = switch_priority(pu.unit)
        prio_d = perspective_eviction_priority(pu)
        marker = ""
        if in_c != in_d:
            marker = " ← DIFFERS"
        notes.append(
            f"  [{pu.idx}] C={'✓' if in_c else '·'} D={'✓' if in_d else '·'} "
            f"op={pu.perspective_op:5s} omega_dir={pu.omega_direction:+.1f} "
            f"ad={pu.anchor_density:.2f}{marker}"
        )

    notes.append("")
    notes.append(f"Resp A: {resp_a[:100]}")
    notes.append(f"Resp C: {resp_c[:100]}")
    notes.append(f"Resp D: {resp_d[:100]}")

    metrics.update({
        "a_all_three": int(score_a["all_three"]),
        "b_all_three": int(score_b["all_three"]),
        "c_all_three": int(score_c["all_three"]),
        "d_all_three": int(score_d["all_three"]),
        "budget":      budget,
        "stp_min":     stp_min,
        "c_units_kept": len(c_kept),
        "d_units_kept": len(d_kept),
    })

    # Pass: D matches or exceeds C
    passed = score_d["all_three"] or (score_d["all_three"] == score_c["all_three"])
    return notes, metrics, passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="qwen2.5:14b")
    parser.add_argument("--stress", type=int, default=None,
                        help="Stress budget below stp_min")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not check_ollama(args.model):
        print(f"[ERROR] Ollama not running or model '{args.model}' not found.")
        sys.exit(1)
    print(f"[OK] Model '{args.model}' found.\n")

    notes, metrics, passed = test_t8_perspective(args.model, args.stress)
    print("\n".join(f"  {n}" for n in notes))
    print()
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()
    print(f"T8-Perspective [{'PASS' if passed else 'FAIL'}]")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
