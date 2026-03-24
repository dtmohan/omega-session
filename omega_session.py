"""
omega_session.py — Generation Layer + Entry Point

Thin wrapper around Ollama that governs context via Ω scoring.
Keeps t/s stable regardless of session length by bounding the fed context.

Usage:
    python omega_session.py
    python omega_session.py --model qwen2.5:14b --budget 3000 --verbose
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from typing import List, Optional

from context_stack import parse_context_units, complexity_counters
from stp_session import (
    open_session, score_unit, build_window, session_report,
    Session, UnitScore,
)


OLLAMA_URL  = "http://localhost:11434/api/chat"
DEFAULT_MODEL  = "qwen2.5:14b"
DEFAULT_BUDGET = 3000   # tokens fed per turn


# ── Ollama client ────────────────────────────────────────────────────────────

def ollama_chat(messages: List[dict], model: str) -> tuple[str, dict]:
    """
    Send messages to Ollama /api/chat.
    Returns (response_text, stats_dict).
    Auditor (scorer) is separate from this Host function. I3.
    """
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise ConnectionError(f"Ollama unreachable at {OLLAMA_URL}: {e}")

    text  = data.get("message", {}).get("content", "")
    stats = {
        "eval_count":    data.get("eval_count", 0),
        "eval_duration": data.get("eval_duration", 1),
        "prompt_tokens": data.get("prompt_eval_count", 0),
    }
    stats["tokens_per_second"] = (
        stats["eval_count"] / (stats["eval_duration"] / 1e9)
        if stats["eval_duration"] > 0 else 0.0
    )
    return text, stats


# ── Session loop ─────────────────────────────────────────────────────────────

class OmegaSessionLoop:
    def __init__(self, model: str, budget_tokens: int, verbose: bool = False):
        self.model         = model
        self.budget_tokens = budget_tokens
        self.verbose       = verbose
        self.history: List[dict] = []   # full conversation, never evicted
        self.session: Optional[Session] = None
        self.turn_index    = 0

    def _init_session(self, first_user_text: str):
        """Declare axis at t=0 from first user message. I5."""
        self.session = open_session(
            axis_text=first_user_text,
            budget_tokens=self.budget_tokens,
        )
        if self.verbose:
            print(f"\n[Session opened] id={self.session.session_id}")
            print(f"[Axis hash]      {self.session.axis_hash}\n")

    def _build_governed_messages(self) -> tuple[List[dict], dict]:
        """
        Score full history, select window, return messages for Ollama.
        Governance precedes the governed event. I1.
        """
        # Parse history into ContextUnits
        turns = [
            {"role": t["role"], "content": t["content"], "turn_index": i}
            for i, t in enumerate(self.history)
        ]
        units = parse_context_units(turns)

        if not units:
            return [], {}

        # Score each unit against session axis
        scores = [
            score_unit(u, self.session, self.turn_index, len(self.history))
            for u in units
        ]

        # Build window within budget
        selected, spent = build_window(
            units, scores, self.budget_tokens, self.turn_index
        )

        # Audit log — append only. I4.
        evicted = len(units) - len(selected)
        self.session.spent_tokens = spent
        self.session.turn_count   = self.turn_index + 1
        self.session.log({
            "event":          "WINDOW_BUILT",
            "turn":           self.turn_index,
            "total_units":    len(units),
            "selected_units": len(selected),
            "evicted_units":  evicted,
            "tokens_fed":     spent,
            "budget":         self.budget_tokens,
        })

        # Reconstruct messages in causal order from selected units
        # Group consecutive units from same turn back together
        turn_texts: dict[tuple, list] = {}
        for u in selected:
            key = (u.turn_index, u.role)
            turn_texts.setdefault(key, []).append(u.text)

        messages = [
            {"role": role, "content": " ".join(texts)}
            for (tidx, role), texts in sorted(turn_texts.items())
        ]

        governance_info = {
            "total_units":    len(units),
            "selected_units": len(selected),
            "tokens_fed":     spent,
            "evicted_units":  len(units) - len(selected),
        }
        return messages, governance_info

    def chat(self, user_input: str) -> str:
        """Single turn. Returns assistant response."""
        # Init session on first turn
        if self.session is None:
            self._init_session(user_input)

        # Store user turn in full history
        self.history.append({"role": "user", "content": user_input})

        # Govern context
        messages, gov_info = self._build_governed_messages()

        if self.verbose:
            print(f"  [Governance] turn={self.turn_index} "
                  f"units={gov_info.get('total_units',0)} "
                  f"selected={gov_info.get('selected_units',0)} "
                  f"evicted={gov_info.get('evicted_units',0)} "
                  f"tokens_fed={gov_info.get('tokens_fed',0)}/{self.budget_tokens}")

        # Generate — Host layer
        t_start = time.time()
        response, stats = ollama_chat(messages, self.model)
        elapsed = time.time() - t_start

        # Store assistant response
        self.history.append({"role": "assistant", "content": response})
        self.turn_index += 1

        if self.verbose:
            tps = stats.get("tokens_per_second", 0)
            print(f"  [Stats] {stats['eval_count']} tokens "
                  f"@ {tps:.1f} t/s in {elapsed:.1f}s\n")

        return response

    def report(self) -> str:
        if self.session:
            return session_report(self.session)
        return "No session active."


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ω-governed Ollama session")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--budget",  type=int, default=DEFAULT_BUDGET,
                        help="Max tokens fed to model per turn")
    parser.add_argument("--verbose", action="store_true",
                        help="Show governance stats per turn")
    args = parser.parse_args()

    loop = OmegaSessionLoop(
        model=args.model,
        budget_tokens=args.budget,
        verbose=args.verbose,
    )

    print(f"Ω Session — model: {args.model} | budget: {args.budget} tokens")
    print("Type 'quit' to exit, 'report' for session stats.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print(loop.report())
            break
        if user_input.lower() == "report":
            print(loop.report())
            continue

        response = loop.chat(user_input)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
