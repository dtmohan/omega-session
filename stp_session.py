"""
stp_session.py — Transport Layer
Axis declaration, Ω scoring, IEP budget enforcement.
Governor enforces SOL → COST → Ω → ALLOW/BRIDGE/FIN.
"""

import hashlib
import time
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from context_stack import ContextUnit, ComplexityCounters


# ── Governor thresholds ──────────────────────────────────────────────────────
SOL_THRESHOLD   = 0.70   # solvency floor
OMEGA_T1        = 0.21   # GREEN  (Ω ≤ T1 → ALLOW)
OMEGA_T2        = 0.30   # YELLOW (T1 < Ω ≤ T2 → BRIDGE-2)
                         # RED    (Ω > T2 → BRIDGE-5 or FIN)

# ── Recency decay ────────────────────────────────────────────────────────────
RECENCY_HALF_LIFE = 10   # turns; older turns decay toward 0.5 weight


@dataclass
class UnitScore:
    unit_hash: str
    turn_index: int
    role: str
    omega: float          # 0.0 = maximally relevant, 1.0 = maximally irrelevant
    sol: float            # solvency of this unit
    tier: str             # CALIBRATED / PROVISIONAL / UNAUDITED
    decision: str         # ALLOW / BRIDGE / FIN
    recency_weight: float
    composite: float      # final keep-score (higher = keep)
    token_estimate: int


@dataclass
class Session:
    session_id: str
    axis_hash: str        # sha256 of unit_0.text — immutable
    axis_text: str
    opened_at: float
    sol_threshold: float
    omega_threshold: float
    budget_tokens: int
    spent_tokens: int = 0
    turn_count: int = 0
    audit_log: List[dict] = field(default_factory=list)
    unit_scores: List[UnitScore] = field(default_factory=list)

    def log(self, event: dict):
        """Append-only audit log. I4."""
        entry = {"t": time.time(), **event}
        self.audit_log.append(entry)


def open_session(
    axis_text: str,
    budget_tokens: int = 4096,
    sol_threshold: float = SOL_THRESHOLD,
    omega_threshold: float = OMEGA_T2,
) -> Session:
    """Declare axis at t=0. Hash-committed. I5."""
    axis_hash  = hashlib.sha256(axis_text.encode()).hexdigest()
    session_id = hashlib.sha256(f"{axis_hash}{time.time()}".encode()).hexdigest()[:12]

    session = Session(
        session_id=session_id,
        axis_hash=axis_hash,
        axis_text=axis_text,
        opened_at=time.time(),
        sol_threshold=sol_threshold,
        omega_threshold=omega_threshold,
        budget_tokens=budget_tokens,
    )
    session.log({
        "event": "SESSION_OPEN",
        "session_id": session_id,
        "axis_hash": axis_hash,
        "budget_tokens": budget_tokens,
    })
    return session


def score_unit(
    unit: ContextUnit,
    session: Session,
    current_turn: int,
    total_turns: int,
) -> UnitScore:
    """
    Compute Ω score for a ContextUnit against the session axis.
    Heuristic v0: structural + recency + role weighting.
    Auditor is architecturally separate from Host. I3.
    """
    # ── Structural Ω (deviation from axis register) ──────────────────────────
    axis_words = set(session.axis_text.lower().split())
    unit_words = set(unit.text.lower().split())
    if not unit_words:
        structural_omega = 1.0
    else:
        overlap = len(axis_words & unit_words) / max(len(unit_words), 1)
        structural_omega = 1.0 - min(overlap * 3.0, 1.0)  # scale: 33% overlap → Ω=0

    # ── Depth bonus (deeper units carry more structural commitment) ───────────
    depth_bonus = min(unit.stack_depth * 0.05, 0.20)
    structural_omega = max(0.0, structural_omega - depth_bonus)

    # ── Recency weight ────────────────────────────────────────────────────────
    turns_ago = current_turn - unit.turn_index
    recency_weight = 0.5 + 0.5 * (2 ** (-turns_ago / RECENCY_HALF_LIFE))

    # ── Role weighting ────────────────────────────────────────────────────────
    role_weight = 1.0 if unit.role == 'user' else 0.85

    # ── Evidence tier ─────────────────────────────────────────────────────────
    if len(unit.text) < 20:
        tier = "UNAUDITED"
        sol  = 0.50
    elif unit.cross_switch:
        tier = "PROVISIONAL"
        sol  = 0.70
    else:
        tier = "CALIBRATED"
        sol  = 0.90

    # ── Governor decision ─────────────────────────────────────────────────────
    omega = structural_omega
    if sol < session.sol_threshold:
        decision = "FIN"
    elif omega <= OMEGA_T1:
        decision = "ALLOW"
    elif omega <= OMEGA_T2:
        decision = "BRIDGE"
    else:
        decision = "BRIDGE" if tier != "UNAUDITED" else "FIN"

    # ── Composite keep-score (higher = higher priority to keep) ───────────────
    # Invert omega so low-omega (relevant) units score high
    composite = (1.0 - omega) * recency_weight * role_weight * sol

    return UnitScore(
        unit_hash=unit.unit_hash,
        turn_index=unit.turn_index,
        role=unit.role,
        omega=omega,
        sol=sol,
        tier=tier,
        decision=decision,
        recency_weight=recency_weight,
        composite=composite,
        token_estimate=unit.token_estimate,
    )


def build_window(
    units: List[ContextUnit],
    scores: List[UnitScore],
    budget_tokens: int,
    current_turn: int,
) -> Tuple[List[ContextUnit], int]:
    """
    Select highest-composite units within token budget.
    Always include the most recent user turn (causality — I1).
    Returns (selected_units_in_order, total_tokens_used).
    """
    # Pin the latest user turn — never evict it
    latest_user_idx = None
    for i in range(len(units) - 1, -1, -1):
        if units[i].role == 'user' and units[i].turn_index == current_turn:
            latest_user_idx = i
            break

    pinned_tokens = units[latest_user_idx].token_estimate if latest_user_idx is not None else 0
    remaining_budget = budget_tokens - pinned_tokens

    # Rank remaining units by composite score
    ranked = sorted(
        [(i, s) for i, s in enumerate(scores) if i != latest_user_idx and s.decision != "FIN"],
        key=lambda x: x[1].composite,
        reverse=True,
    )

    selected_indices = set()
    if latest_user_idx is not None:
        selected_indices.add(latest_user_idx)

    spent = pinned_tokens
    for idx, score in ranked:
        if spent + score.token_estimate <= remaining_budget:
            selected_indices.add(idx)
            spent += score.token_estimate

    # Return in original order (preserve causality)
    selected = [units[i] for i in sorted(selected_indices)]
    return selected, spent


def session_report(session: Session) -> str:
    lines = [
        f"Session: {session.session_id}",
        f"Axis:    {session.axis_hash}",
        f"Turns:   {session.turn_count}",
        f"Budget:  {session.spent_tokens}/{session.budget_tokens} tokens",
        f"Log entries: {len(session.audit_log)}",
    ]
    return "\n".join(lines)
