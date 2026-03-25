"""
test_t8_sonnet.py — T8 Cross-Model Comparison

Runs the same three conditions from test_t8.py against Claude Sonnet
via the Anthropic API, to isolate whether Condition A's failure is:

  (a) Model capability: small models confabulate known philosophers
      from training rather than reading context carefully
  (b) Attention architecture: recency bias in how models weight
      early vs late context regardless of capability
  (c) Both

If Sonnet passes Condition A:
    → capability problem. Qwen confabulates; Sonnet reads context.
    → the fix for production is a capable model, not a governance layer.

If Sonnet fails Condition A:
    → architecture problem. Even the best models weight recent context
      more heavily. Governance (C) is required regardless of model.
    → the fix is structural ordering, not model size.

The hypothesis (from security primitive paper + T8 analysis):
    Condition A fails for both models, but for different reasons.
    Qwen fails partly by confabulation, partly by recency bias.
    Sonnet fails only by recency bias — it will read whatever is
    in context correctly, but it still weights recent context higher.

Run:
    python test_t8_sonnet.py
    python test_t8_sonnet.py --stress 400
"""

import os
import sys
import time
import json
import argparse
import urllib.request
import urllib.error

sys.path.insert(0, ".")
from document_segmenter import parse_context_units, complexity_counters, build_axis_stack

# ── Essay + planted fact (same as test_t8.py) ────────────────────────────────
# Imported to ensure identical corpus across both test files
from test_t8 import (
    PHILOSOPHICAL_ESSAY,
    PLANTED_FACT,
    RECALL_PROBE,
    RECALL_KEYWORDS,
    recall_score,
)

SONNET_MODEL = "claude-sonnet-4-20250514"
API_URL      = "https://api.anthropic.com/v1/messages"

# ── API key ───────────────────────────────────────────────────────────────────
# Set in your shell before running:
#   export ANTHROPIC_API_KEY="sk-ant-..."
# Or in a .env file (loaded below if python-dotenv is installed).
# Never hardcode the key in this file.

def _get_api_key() -> str:
    # 1. Check environment variable first
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key

    # 2. Try loading from .env file in the same directory as this script
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key

    # 3. Try ~/.anthropic/api_key (common local storage location)
    home_key_path = os.path.expanduser("~/.anthropic/api_key")
    if os.path.exists(home_key_path):
        with open(home_key_path) as f:
            key = f.read().strip()
        if key:
            return key

    print("[ERROR] ANTHROPIC_API_KEY not found. Set it one of three ways:")
    print()
    print("  1. Environment variable (this session only):")
    print("       export ANTHROPIC_API_KEY='sk-ant-...'")
    print()
    print("  2. .env file in omega-session/:")
    print("       echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env")
    print()
    print("  3. ~/.anthropic/api_key file:")
    print("       mkdir -p ~/.anthropic")
    print("       echo 'sk-ant-...' > ~/.anthropic/api_key")
    print("       chmod 600 ~/.anthropic/api_key")
    sys.exit(1)


# ── Anthropic API call ────────────────────────────────────────────────────────

def sonnet_chat(messages: list, system: str = None) -> str:
    """
    Single call to claude-sonnet via Anthropic API.
    Returns the assistant text response.
    """
    api_key = _get_api_key()

    payload = {
        "model":      SONNET_MODEL,
        "max_tokens": 512,
        "messages":   messages,
    }
    if system:
        payload["system"] = system

    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        API_URL,
        data    = data,
        headers = {
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method = "POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"[API ERROR] HTTP {e.code}: {body[:200]}")
        raise

    # Extract text from content blocks
    for block in resp.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return ""


# ── Shared system prompt ─────────────────────────────────────────────────────
# Explicit instruction to read from context, not from training knowledge.
# This isolates whether capability (following instructions) or architecture
# (attention weighting) is the failure mode.

SYSTEM_PROMPT = (
    "You are a careful reading assistant. When asked to recall information, "
    "you ONLY report what was explicitly stated in the conversation above. "
    "You do NOT supplement with your own knowledge. If something was not "
    "stated in the prior messages, say so explicitly rather than guessing."
)

SYSTEM_PROMPT_BLIND = None  # No explicit instruction — baseline


# ── Condition A: Sliding window (recency order) ───────────────────────────────

def run_condition_a_sonnet(units, budget_tokens: int, use_system: bool = True) -> dict:
    """
    All units that fit in budget, ordered by recency (last units closest to probe).
    Tests whether Sonnet's recency bias matches Qwen's.
    """
    all_messages = []
    for u in units:
        all_messages.append({"role": "user",      "content": u.text[:1200]})
        all_messages.append({"role": "assistant", "content": "Understood."})

    # Sliding window from the end
    kept         = []
    budget_chars = budget_tokens * 4
    total_chars  = 0
    for msg in reversed(all_messages):
        c = len(msg["content"])
        if total_chars + c > budget_chars:
            break
        kept.insert(0, msg)
        total_chars += c

    kept.append({"role": "user", "content": RECALL_PROBE})
    system   = SYSTEM_PROMPT if use_system else SYSTEM_PROMPT_BLIND
    response = sonnet_chat(kept, system)

    return {
        "response":    response,
        "turns_kept":  len(kept) // 2,
        "scores":      recall_score(response),
        "system_used": use_system,
    }


# ── Condition A': Axis-first ordering ────────────────────────────────────────

def run_condition_a_prime_sonnet(units, budget_tokens: int, use_system: bool = True) -> dict:
    """
    Same units as A, but ordered with axis (unit 0) FIRST regardless of recency.
    If A' passes where A fails → purely a positional/ordering problem.
    If A' also fails → deeper attention weighting issue.
    """
    all_messages = []
    for u in units:
        all_messages.append({"role": "user",      "content": u.text[:1200]})
        all_messages.append({"role": "assistant", "content": "Understood."})

    # Select units that fit in budget (from recency end, same as A)
    kept_raw     = []
    budget_chars = budget_tokens * 4
    total_chars  = 0
    for msg in reversed(all_messages):
        c = len(msg["content"])
        if total_chars + c > budget_chars:
            break
        kept_raw.insert(0, msg)
        total_chars += c

    # Reorder: axis (unit 0 = first two messages) goes first
    # The rest stay in their original order after axis
    axis_msgs = all_messages[:2]  # unit 0 user + "Understood."
    rest      = [m for m in kept_raw if m not in axis_msgs]
    # Ensure axis is in kept
    if axis_msgs[0] not in kept_raw:
        rest = kept_raw  # axis wasn't selected — add it first anyway
    ordered = axis_msgs + rest

    ordered.append({"role": "user", "content": RECALL_PROBE})
    system   = SYSTEM_PROMPT if use_system else SYSTEM_PROMPT_BLIND
    response = sonnet_chat(ordered, system)

    return {
        "response":    response,
        "turns_kept":  len(ordered) // 2,
        "axis_first":  True,
        "scores":      recall_score(response),
        "system_used": use_system,
    }


# ── Condition C: Depth-stratified (same logic as test_t8.py) ─────────────────

def run_condition_c_sonnet(units, budget_tokens: int, use_system: bool = True) -> dict:
    """
    Depth-stratified selection + axis-first ordering, fed to Sonnet.
    This is the control: if C passes for Sonnet at tight budget while A fails,
    the governance layer adds value beyond model capability.
    """
    axis_result    = build_axis_stack(units)
    surface_audits = axis_result.get("surface_audits", [])
    decision_map   = {}
    for a in surface_audits:
        idx = a.get("unit_idx")
        if idx is not None:
            decision_map[idx] = a.get("decision", "ALLOW")

    def unit_priority(u):
        if u.idx == 0:            return 0
        decision = decision_map.get(u.idx, "ALLOW")
        if decision == "FIN":     return 4
        if u.cross_switch:        return 3
        if u.switch_index > 0:    return 2
        if decision == "NACK":    return 3
        return 1

    budget_chars  = budget_tokens * 4
    sorted_units  = sorted(units, key=unit_priority)
    kept_messages = []
    total_chars   = 0
    for u in sorted_units:
        c = min(len(u.text), 1200)
        if total_chars + c > budget_chars:
            continue
        kept_messages.append({
            "content":   u.text[:1200],
            "_unit_idx": u.idx,
            "_priority": unit_priority(u),
        })
        total_chars += c

    kept_messages.sort(key=lambda m: m["_unit_idx"])

    clean_messages = []
    for m in kept_messages:
        clean_messages.append({"role": "user",      "content": m["content"]})
        clean_messages.append({"role": "assistant", "content": "Understood."})
    clean_messages.append({"role": "user", "content": RECALL_PROBE})

    system   = SYSTEM_PROMPT if use_system else SYSTEM_PROMPT_BLIND
    response = sonnet_chat(clean_messages, system)

    pivot_kept  = sum(1 for m in kept_messages if m["_priority"] >= 2)
    axis_kept   = sum(1 for m in kept_messages if m["_priority"] <= 1)
    pivot_count = sum(1 for u in units if u.switch_index > 0)

    return {
        "response":      response,
        "units_kept":    len(kept_messages),
        "axis_kept":     axis_kept,
        "pivot_count":   pivot_count,
        "pivot_kept":    pivot_kept,
        "eviction_rate": 1.0 - (len(kept_messages) / max(len(units), 1)),
        "scores":        recall_score(response),
        "system_used":   use_system,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_sonnet_t8(stress_budget: int = None, verbose: bool = False):
    print(f"\n{'='*60}")
    print("T8 Cross-Model: Claude Sonnet")
    mode = f"stress={stress_budget} tok" if stress_budget else "IEP-dynamic"
    print(f"Model: {SONNET_MODEL}  |  Mode: {mode}")
    print(f"{'='*60}\n")

    # Segment
    print("Segmenting essay...")
    all_units = parse_context_units(PHILOSOPHICAL_ESSAY)
    units     = [u for u in all_units if len(u.text) // 4 >= 20]
    for i, u in enumerate(units):
        u.idx = i

    counters     = complexity_counters(units)
    total_tokens = sum(len(u.text) // 4 for u in units)
    depth        = counters.get("stack_depth", 0)
    switches     = counters.get("context_switches", 0)
    iep_min      = 500 + (depth * 200) + (switches * 100)
    axis_tok     = len(units[0].text) // 4

    budget_tokens = stress_budget if stress_budget else iep_min

    print(f"  Units: {len(units)}  |  IEP min: {iep_min} tok  |  Axis: {axis_tok} tok")
    print(f"  Budget: {budget_tokens} tok  "
          f"({'STRESS' if stress_budget else 'IEP-derived'})")
    if stress_budget and stress_budget < axis_tok:
        print(f"  !! INVALID: budget < axis unit — all conditions fail by construction")
    print()

    if verbose:
        for u in units:
            toks   = len(u.text) // 4
            sw_tag = "[PIVOT]" if u.switch_index > 0 else "[AXIS] "
            print(f"  [{u.idx}] sw={u.switch_index} ~{toks}tok {sw_tag}  "
                  f"'{u.text[:60].replace(chr(10),' ').strip()}'")
        print()

    print("Planted thesis:", PLANTED_FACT)
    print()

    # ── Run conditions ────────────────────────────────────────────────────────
    results = {}

    # A: sliding window, recency order, WITH explicit system prompt
    print("Running A  (sliding, recency, with system prompt)...")
    t0 = time.time()
    results["A_guided"] = run_condition_a_sonnet(units, budget_tokens, use_system=True)
    print(f"  [{time.time()-t0:.1f}s]")

    # A (blind): same but WITHOUT system prompt — pure model behavior
    print("Running A  (sliding, recency, NO system prompt)...")
    t0 = time.time()
    results["A_blind"] = run_condition_a_sonnet(units, budget_tokens, use_system=False)
    print(f"  [{time.time()-t0:.1f}s]")

    # A': axis-first ordering, WITH system prompt
    print("Running A' (axis-first ordering, with system prompt)...")
    t0 = time.time()
    results["A_prime"] = run_condition_a_prime_sonnet(units, budget_tokens, use_system=True)
    print(f"  [{time.time()-t0:.1f}s]")

    # C: depth-stratified, WITH system prompt
    print("Running C  (depth-stratified, with system prompt)...")
    t0 = time.time()
    results["C"] = run_condition_c_sonnet(units, budget_tokens, use_system=True)
    print(f"  [{time.time()-t0:.1f}s]")

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    print("─" * 60)

    def fmt(s):
        return (f"Lewis={s['lewis']}  Armstrong={s['armstrong']}  "
                f"Shoemaker={s['shoemaker']}  all_three={s['all_three']}")

    labels = {
        "A_guided": "A  (sliding, recency, guided)",
        "A_blind":  "A  (sliding, recency, blind) ",
        "A_prime":  "A' (axis-first, guided)      ",
        "C":        "C  (depth-stratified, guided)",
    }

    for key, label in labels.items():
        r = results[key]
        score_str = fmt(r["scores"])
        extra = ""
        if "eviction_rate" in r:
            extra = f"  eviction={r['eviction_rate']:.0%}"
        elif "turns_kept" in r:
            extra = f"  turns_kept={r['turns_kept']}"
        print(f"{label}:  {score_str}{extra}")

    print()
    print("Responses (first 150 chars each):")
    for key, label in labels.items():
        print(f"  {label}:")
        print(f"    {results[key]['response'][:150]}")
        print()

    print("─" * 60)
    print()

    # ── Diagnosis ────────────────────────────────────────────────────────────
    a_guided = results["A_guided"]["scores"]["all_three"]
    a_blind  = results["A_blind"]["scores"]["all_three"]
    a_prime  = results["A_prime"]["scores"]["all_three"]
    c_pass   = results["C"]["scores"]["all_three"]

    # Score counts (not just all_three — partial hits matter for diagnosis)
    def score_count(r): return sum([r["lewis"], r["armstrong"], r["shoemaker"]])
    sc_a_guided = score_count(results["A_guided"]["scores"])
    sc_a_blind  = score_count(results["A_blind"]["scores"])
    sc_a_prime  = score_count(results["A_prime"]["scores"])
    sc_c        = score_count(results["C"]["scores"])

    print("Score counts (out of 3):")
    print(f"  A  guided:  {sc_a_guided}/3")
    print(f"  A  blind:   {sc_a_blind}/3")
    print(f"  A' guided:  {sc_a_prime}/3")
    print(f"  C  guided:  {sc_c}/3")
    print()

    print("Diagnosis:")

    # Detect guided-worse-than-blind (the backfire)
    if sc_a_guided < sc_a_blind:
        print(f"  !! System prompt BACKFIRE: guided={sc_a_guided}/3 < blind={sc_a_blind}/3")
        print("     Conservative retrieval instruction ('only report from context')")
        print("     compounded recency bias. Sonnet said 'not in context' rather than")
        print("     searching the early window. Blind condition found partial signal")
        print("     precisely because it had no instruction lowering the reporting threshold.")
        print("     Operational finding: explicit retrieval constraints can make recency")
        print("     bias worse by giving the model permission to under-report.")
    elif sc_a_guided > sc_a_blind:
        print(f"  ✓ System prompt helped: guided={sc_a_guided}/3 > blind={sc_a_blind}/3")
        print("     Sonnet's conservatism ('only report from context') compounded")
        print("     recency bias: it said 'I don't see it' rather than finding it.")
        print("     Operational finding: explicit retrieval instructions can backfire")
        print("     when positional weighting is already causing underrepresentation.")
    elif a_blind and not a_guided:
        print("  UNUSUAL: blind passed, guided failed — system prompt interfered.")
    elif not a_blind and a_guided:
        print("  ✓ System prompt helped — model follows explicit retrieval instructions.")
    elif a_blind and a_guided:
        print("  ✓ Sonnet passes A regardless of system prompt — capability overcomes")
        print("    recency bias. This is a capability problem, not architecture.")
    else:
        # both fail
        print("  Both A variants fail — recency bias persists even for capable model.")
        print("  This is an attention architecture problem, not a capability problem.")

    # A vs A'
    if not a_guided and a_prime:
        print("  ✓ A' (axis-first) passes where A fails — purely a positional problem.")
        print("    Axis-first ordering is the minimum viable fix for recency bias.")
        print("    No selection governance needed at adequate budget — just placement.")
    elif not a_guided and not a_prime:
        print("  A' also fails — ordering alone doesn't fix it.")
        print("    Attention weighting degrades for early context even when placed first.")
    elif a_guided and a_prime:
        print("  Both A and A' pass — no ordering sensitivity at this budget level.")

    # C vs A
    if not a_guided and c_pass:
        print("  ✓ C passes where A fails — depth-stratified governance adds real value")
        print("    beyond model capability, even for Sonnet. Solves both selection")
        print("    (budget pressure) and placement (axis always first in output).")

    print()
    passed = c_pass
    print(f"T8-Sonnet [{'PASS' if passed else 'FAIL'}]")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress",  type=int, default=None,
                        help="Stress test with fixed budget below IEP minimum")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_sonnet_t8(args.stress, args.verbose)


if __name__ == "__main__":
    main()
