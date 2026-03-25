"""
context_stack.py
═══════════════════════════════════════════════════════════════
Context Stack Primitive

One implementation. Two consumers:
  - rstp_ingestion._fast_pass(): segments documents into context units
  - audit/engine.py: scores context units, not character windows

The O(1) principle (from FINAL_ARCHITECTURE_OPTIMIZED):
  stack_depth      = qualifier nesting depth       (how deep are we?)
  context_switches = running switch count          (how many transitions?)
  estimated_window = base + (depth*200) + (switches*100)

A "because" is only a closure violation if it bridges ACROSS a context
switch — borrowing causal force from a prior unit without re-establishing
license. Inside a single unit, the whole unit is evaluated together.

"Not because X. But because Y" — same unit, same depth, contrastive
completion of one rhetorical move. No switch fires. Not a violation.

Stack operations:
  PUSH    — qualifier opens a new nested context
            ("if", "unless", "except", "provided that", "assuming")
  POP     — qualifier resolves, return to parent
            ("then", "therefore", "thus", "hence", "so", "consequently")
  SWITCH  — lateral move at same depth, new subject or contrastive pivot
            ("however", "but", "yet", "nevertheless", "on the other hand",
             "by contrast", "alternatively", "in contrast")
  HOLD    — same context continues (most sentences)

Subject change detection (triggers SWITCH at same depth):
  - New named entity takes over as grammatical subject
  - Section/heading marker (##, ---, etc.)
  - Sentence-initial "The [new noun]..." after a different subject chain

═══════════════════════════════════════════════════════════════
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ── Marker tables ──────────────────────────────────────────────────────────────

# PUSH: opens a new nested context — increases depth
QUALIFIER_PUSH = re.compile(
    r'\b(if|unless|except(?:\s+when|\s+where|\s+that)?|'
    r'provided\s+that|assuming\s+that|assuming|'
    r'suppose(?:d\s+that)?|in\s+the\s+event\s+that|'
    r'whenever|wherever|whether)\b',
    re.IGNORECASE
)

# POP: resolves a qualifier — decreases depth, returns to parent
QUALIFIER_POP = re.compile(
    r'(?:^|\.\s+|\n)(?:then|therefore|thus|hence|'
    r'consequently|as\s+a\s+result|it\s+follows|'
    r'so\b|and\s+so\b)',
    re.IGNORECASE
)

# SWITCH: lateral pivot at same depth — increments switch counter
CONTEXT_PIVOT = re.compile(
    r'(?:^|(?<=\.)\s+|\n)'
    r'(however|but\s+(?!because\b)|yet\s+(?!again\b)|'
    r'nevertheless|nonetheless|on\s+the\s+other\s+hand|'
    r'by\s+contrast|in\s+contrast|alternatively|'
    r'that\s+said|even\s+so|still\b|and\s+yet\b)',
    re.IGNORECASE
)

# Section boundary: heading or horizontal rule — always a SWITCH
SECTION_BOUNDARY = re.compile(
    r'(?:^|\n)(#{1,6}\s+.+|[-─═]{3,}|[*]{3,})\s*(?:\n|$)',
    re.MULTILINE
)




# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ContextUnit:
    """
    A coherent context segment between switches.

    Scored as a whole by the audit engine.
    Causal bridges that stay inside this unit are evaluated
    in full context — not flagged in isolation.
    """
    idx:             int
    text:            str
    char_start:      int
    char_end:        int
    stack_depth:     int        # Nesting depth at time of unit creation
    switch_index:    int        # Which switch produced this unit
    push_count:      int        # Qualifiers opened inside this unit
    pop_count:       int        # Qualifiers resolved inside this unit
    cross_switch:    bool       # True if a causal bridge spans into prior unit
    residue_in:     float = 0.0  # Cpigment carried in from prior Sandhi releases
    anchor_distance: int  = 0    # tokens since last anchor at unit start




# ── Canonical STP phased-return constant (governor.py l_rate default) ────────
# S_{t+1} = (1 - STP_LAMBDA) * S_t + STP_LAMBDA * V_0
# Replaces arbitrary 0.6^age exponential decay.
# Source: STPKernel.apply_phased_return(), bridge_tax_config.return_policy.lambda
STP_LAMBDA: float = 0.25

@dataclass
class AxisSlice:
    """
    One slice of the depth-stratified axis.

    Built during descent: each unit at this depth contributes to the slice.
    Consumed during ascent: surfacing content is audited against this slice.

    The whole governance design is invoked at each surface event,
    but anchored to the axis built at that depth — not the global t=0.

    This is the recursive invariant:
        same mechanism, different context.
    """
    depth:          int
    units:          list        # ContextUnit indices that built this slice
    omega_sum:      float = 0.0 # accumulated pressure at this depth
    unit_count:     int   = 0
    committed:      bool  = False  # True once first pop from this depth fires

    @property
    def omega_mean(self) -> float:
        return self.omega_sum / max(1, self.unit_count)

    def add_unit(self, unit_idx: int, pressure: float = 0.0):
        self.units.append(unit_idx)
        self.omega_sum += pressure
        self.unit_count += 1

    def audit_against(self, surface_pressure: float) -> dict:
        """
        Governance decision: surface content vs axis built at this depth.
        Returns ALLOW / NACK / FIN with distance from depth axis.
        """
        distance = abs(surface_pressure - self.omega_mean)
        if distance > 0.7:
            decision = "FIN"
        elif distance > 0.4:
            decision = "NACK"
        else:
            decision = "ALLOW"
        return {
            "decision":       decision,
            "distance":       round(distance, 4),
            "axis_mean":      round(self.omega_mean, 4),
            "surface":        round(surface_pressure, 4),
            "depth":          self.depth,
        }

@dataclass
class SandhiResidue:
    """
    Pressure residue emitted at each Sandhi point.

    The IEP principle: at every context release — pop, switch, section,
    attribution drop, register break, closure signal — the accumulated
    pressure from the releasing frame must be carried forward into the
    parent/successor frame as a weighted residue.

    Sn+1 = f(Sn, Cpigment) + delta_interaction

    Cpigment is the Origin Vector pressure. It cannot be dropped.
    It decays with depth and distance from origin, but it persists.

    Fields:
      sandhi_type     — what kind of release produced this residue
      omega_at_release — Ω accumulated at the moment of release
      depth_at_release — nesting depth when released
      switch_at_release — switch count when released
      anchor_distance  — tokens since last attributed claim (anchor drift)
      decay            — computed decay factor: 1/(1 + depth)
      weight           — omega_at_release * decay (carries into parent)
      unit_idx         — which unit emitted this residue
    """
    sandhi_type:       str    # POP | SWITCH | SECTION | ATTRIB_DROP | CLOSURE
    omega_at_release:  float
    depth_at_release:  int
    switch_at_release: int
    anchor_distance:   int    # tokens since last anchored claim
    unit_idx:          int
    decay:             float  = 0.0
    weight:            float  = 0.0

    def __post_init__(self):
        # IEP decay: deeper frames contribute less residue to parent
        # A frame at depth 0 contributes fully; depth 3 contributes 25%
        self.decay  = 1.0 / (1.0 + self.depth_at_release)
        self.weight = round(self.omega_at_release * self.decay, 4)


@dataclass
class StackState:
    """Running state of the context stack parser."""
    depth:          int   = 0
    switches:       int   = 0
    last_subject:   str   = ""
    unit_start:     int   = 0
    push_count:     int   = 0
    pop_count:      int   = 0
    anchor_distance: int  = 0    # tokens since last attributed claim
    omega:          float = 0.0  # running Ω estimate (rough, for residue seeding)
    residues:       list  = field(default_factory=list)  # SandhiResidue list


# ── Core primitive ─────────────────────────────────────────────────────────────

def parse_context_units(text: str) -> List[ContextUnit]:
    """
    Stream text, detect context switches, emit context units.

    This is the O(1)-per-token primitive. The switch counter
    is the structural complexity metric. Each unit is a coherent
    segment the audit engine scores as a whole.

    Returns: ordered list of ContextUnit
    """
    units: List[ContextUnit] = []
    state = StackState()
    pos = 0
    n = len(text)

    # Collect all switch positions first (one pass)
    switch_positions = _find_switch_positions(text)

    # If no switches found, the whole text is one unit
    if not switch_positions:
        return [ContextUnit(
            idx=0, text=text.strip(),
            char_start=0, char_end=n,
            stack_depth=0, switch_index=0,
            push_count=_count_matches(QUALIFIER_PUSH, text),
            pop_count=_count_matches(QUALIFIER_POP, text),
            cross_switch=False
        )]

    # Emit units at each switch boundary
    unit_idx = 0
    prev_end = 0

    for sw_pos, sw_type, sw_depth_delta in switch_positions:
        unit_text = text[prev_end:sw_pos].strip()
        if unit_text:
            units.append(ContextUnit(
                idx=unit_idx,
                text=unit_text,
                char_start=prev_end,
                char_end=sw_pos,
                stack_depth=state.depth,
                switch_index=state.switches,
                push_count=_count_matches(QUALIFIER_PUSH, unit_text),
                pop_count=_count_matches(QUALIFIER_POP, unit_text),
                cross_switch=False,     # Updated in cross-switch analysis
                anchor_distance=state.anchor_distance,
            ))
            unit_idx += 1

        # Emit Sandhi residue BEFORE applying the state change
        # The residue carries the pressure accumulated in the releasing frame
        if units and sw_type in ("POP", "SWITCH", "SECTION"):
            # Estimate omega from cross_switch density of emitted units so far
            recent_units = units[-3:] if len(units) >= 3 else units
            omega_est = sum(
                0.3 if u.cross_switch else 0.05
                for u in recent_units
            ) / max(1, len(recent_units))
            residue = SandhiResidue(
                sandhi_type      = sw_type,
                omega_at_release = omega_est + state.omega,
                depth_at_release = state.depth,
                switch_at_release= state.switches,
                anchor_distance  = state.anchor_distance,
                unit_idx         = unit_idx - 1 if unit_idx > 0 else 0,
            )
            state.residues.append(residue)

        # Apply stack operation
        if sw_type == "PUSH":
            state.depth = max(0, state.depth + 1)
            state.anchor_distance += 1   # depth increase = moving away from anchor
        elif sw_type == "POP":
            state.depth = max(0, state.depth - 1)
        elif sw_type == "SWITCH":
            state.switches += 1
            state.anchor_distance += 1   # lateral switch drifts from anchor
        elif sw_type == "SECTION":
            state.switches += 1
            state.depth = 0              # Section boundary resets depth
            state.anchor_distance = 0    # Section is a hard anchor reset

        prev_end = sw_pos

    # Final unit: text after last switch
    final_text = text[prev_end:].strip()
    if final_text:
        units.append(ContextUnit(
            idx=unit_idx,
            text=final_text,
            char_start=prev_end,
            char_end=n,
            stack_depth=state.depth,
            switch_index=state.switches,
            push_count=_count_matches(QUALIFIER_PUSH, final_text),
            pop_count=_count_matches(QUALIFIER_POP, final_text),
            cross_switch=False,
            anchor_distance=state.anchor_distance,
        ))

    # Mark units that begin with cross-switch causal markers
    _mark_cross_switch_units(units)

    # Propagate Sandhi residues — IEP Cpigment carry-forward
    _propagate_sandhi_residues(units)

    return units


def complexity_counters(units: List[ContextUnit]) -> dict:
    """
    Return O(1) complexity counters for the full document.

    These are the structural metrics from FINAL_ARCHITECTURE_OPTIMIZED:
      stack_depth      = max nesting depth seen
      context_switches = total lateral switches
      estimated_window = base + (depth*200) + (switches*100)
    """
    if not units:
        return {"stack_depth": 0, "context_switches": 0,
                "estimated_window": 500, "unit_count": 0}

    max_depth   = max(u.stack_depth for u in units)
    switches    = units[-1].switch_index  # Running counter at last unit
    est_window  = 500 + (max_depth * 200) + (switches * 100)

    # Collect all residues from units (stored on first unit via propagation)
    all_residues = getattr(units[0], "_residues", []) if units else []
    total_residue_weight = sum(r.weight for r in all_residues)
    max_anchor_distance  = max((u.anchor_distance for u in units), default=0)

    # IEP extended estimated_window:
    # base + depth*200 + switches*100 + residue*150 + anchor_drift*50
    est_window = (500
        + (max_depth * 200)
        + (switches * 100)
        + (total_residue_weight * 150)
        + (max_anchor_distance * 50))

    return {
        "stack_depth":           max_depth,
        "context_switches":      switches,
        "estimated_window":      est_window,
        "unit_count":            len(units),
        "total_residue_weight":  round(total_residue_weight, 4),
        "max_anchor_distance":   max_anchor_distance,
        "residue_count":         len(all_residues),
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _find_switch_positions(text: str) -> List[Tuple[int, str, int]]:
    """
    Find all context switch positions in one pass.
    Returns list of (char_position, switch_type, depth_delta).

    switch_type: "PUSH" | "POP" | "SWITCH" | "SECTION"
    depth_delta: +1 (push), -1 (pop), 0 (switch/section)
    """
    events = []

    # Section boundaries first (highest priority)
    for m in SECTION_BOUNDARY.finditer(text):
        events.append((m.start(), "SECTION", 0))

    # Qualifier push (nesting opens)
    # Accept sentence-initial, after punctuation, OR after em-dash (mid-clause nesting)
    for m in QUALIFIER_PUSH.finditer(text):
        pre = text[max(0, m.start()-5):m.start()].strip()
        if pre in ('', ',', ';', '.', '?', '!') or pre.endswith(('—', '–', '-', ',')):
            events.append((m.start(), "PUSH", +1))

    # Qualifier pop (nesting resolves)
    for m in QUALIFIER_POP.finditer(text):
        events.append((m.start(), "POP", -1))

    # Lateral pivots
    for m in CONTEXT_PIVOT.finditer(text):
        events.append((m.start(), "SWITCH", 0))

    # Sort by position, deduplicate overlapping
    events.sort(key=lambda e: e[0])
    events = _deduplicate_events(events)

    return events




def _deduplicate_events(events: list) -> list:
    """
    Remove overlapping events — keep highest-priority type.
    Priority: SECTION > PUSH > POP > SWITCH
    Minimum gap between events: 10 chars.
    """
    if not events:
        return events

    priority = {"SECTION": 0, "PUSH": 1, "POP": 2, "SWITCH": 3}
    deduped = [events[0]]

    for ev in events[1:]:
        last = deduped[-1]
        if ev[0] - last[0] < 10:
            # Too close — keep higher priority
            if priority[ev[1]] < priority[last[1]]:
                deduped[-1] = ev
        else:
            deduped.append(ev)

    return deduped


def _count_matches(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))


def _mark_cross_switch_units(units: List[ContextUnit]):
    """
    Mark units that open with a causal bridge (because/therefore/thus)
    immediately after a switch boundary.

    These are the propagation violations: borrowing causal force
    from a prior context unit that was established under different
    subject/qualifier conditions.
    """
    CROSS_SWITCH_CAUSAL = re.compile(
        r'^\s*(because|therefore|thus|hence|consequently|'
        r'as\s+a\s+result|for\s+this\s+reason|'
        r'that\s+is\s+why|which\s+is\s+why)\b',
        re.IGNORECASE
    )

    for i, unit in enumerate(units):
        if i == 0:
            continue
        # Check if this unit starts with a causal bridge
        if CROSS_SWITCH_CAUSAL.match(unit.text):
            unit.cross_switch = True


def _propagate_sandhi_residues(units: List[ContextUnit]):
    """
    IEP Cpigment carry-forward: propagate Sandhi residues into successor units.

    Derives release events from unit sequence deltas — no external state needed:
      - switch_index increased   → SWITCH event between units
      - stack_depth decreased    → POP event between units
      - stack_depth > 0          → we are inside a nested frame
      - cross_switch             → causal bridge borrowed from prior context
      - anchor_distance high     → attribution drift from Origin Vector

    Decay: phased return S_{t+1}=(1-λ)·S_t + λ·V_0, λ=STP_LAMBDA=0.25 (canonical).
    Pruning: residues < 0.01 are dropped.

    stores _residues on units[0] for complexity_counters.
    """
    if not units:
        return

    accumulated: list = []   # active SandhiResidue objects
    all_emitted: list = []   # for complexity_counters

    for i, unit in enumerate(units):
        # ── Seed this unit from active residues ───────────────────────────────
        active_weight = 0.0
        still_active  = []
        for r in accumulated:
            age     = i - r.unit_idx
            # Canonical phased return: S_{t+1} = (1-λ)·S_t + λ·V_0
            # V_0 = 0.0 (origin is clean state), closed form: (1-λ)^age * S_0
            # λ = STP_LAMBDA = 0.25  (from governor.py bridge_tax_config)
            decayed = r.weight * ((1.0 - STP_LAMBDA) ** age)
            if decayed > 0.01:
                active_weight += decayed
                still_active.append(r)
        accumulated   = still_active
        unit.residue_in = round(active_weight, 4)

        # ── Detect release events from unit deltas ────────────────────────────
        if i == 0:
            continue

        prev = units[i - 1]
        depth_delta  = unit.stack_depth - prev.stack_depth
        switch_delta = unit.switch_index - prev.switch_index

        # POP: depth decreased — child frame closed, carry its pressure up
        if depth_delta < 0:
            omega_est = max(0.1, prev.stack_depth * 0.15 + prev.anchor_distance * 0.02)
            r = SandhiResidue(
                sandhi_type      = "POP",
                omega_at_release = omega_est,
                depth_at_release = prev.stack_depth,
                switch_at_release= prev.switch_index,
                anchor_distance  = prev.anchor_distance,
                unit_idx         = i,
            )
            accumulated.append(r)
            all_emitted.append(r)

        # SWITCH: lateral pivot — prior subject pressure carries forward
        if switch_delta > 0:
            cross_weight = 0.3 if prev.cross_switch else 0.08
            omega_est    = cross_weight + (prev.anchor_distance * 0.02)
            r = SandhiResidue(
                sandhi_type      = "SWITCH",
                omega_at_release = omega_est,
                depth_at_release = prev.stack_depth,
                switch_at_release= prev.switch_index,
                anchor_distance  = prev.anchor_distance,
                unit_idx         = i,
            )
            accumulated.append(r)
            all_emitted.append(r)

        # ATTRIB_DROP: this unit opens with a cross-switch causal bridge
        # — it borrows causal force from a prior context without re-establishing
        if unit.cross_switch:
            r = SandhiResidue(
                sandhi_type      = "ATTRIB_DROP",
                omega_at_release = 0.4 + (unit.anchor_distance * 0.02),
                depth_at_release = unit.stack_depth,
                switch_at_release= unit.switch_index,
                anchor_distance  = unit.anchor_distance,
                unit_idx         = i,
            )
            accumulated.append(r)
            all_emitted.append(r)

        # ANCHOR_DRIFT: anchor distance climbed — Origin Vector drifting
        if unit.anchor_distance > 4 and prev.anchor_distance <= 4:
            r = SandhiResidue(
                sandhi_type      = "ANCHOR_DRIFT",
                omega_at_release = min(0.5, unit.anchor_distance * 0.04),
                depth_at_release = unit.stack_depth,
                switch_at_release= unit.switch_index,
                anchor_distance  = unit.anchor_distance,
                unit_idx         = i,
            )
            accumulated.append(r)
            all_emitted.append(r)

    # Store for complexity_counters
    units[0].__dict__["_residues"] = all_emitted




def build_axis_stack(units: list) -> dict:
    """
    Build the depth-stratified axis from the unit sequence.

    Descent: units accumulate into their depth's AxisSlice.
    Ascent: each pop/switch fires governance against the slice built at that depth.

    The recursive invariant — "the whole design invoked at a different context" —
    is implemented here: governance fires at every surface event with the axis
    that was built during the descent to that depth.

    Returns:
        {
          "slices":        {depth: AxisSlice},
          "surface_audits": [audit_result_dict, ...],  # one per surface event
          "max_depth":     int,
          "total_surfaces": int,
        }
    """
    if not units:
        return {"slices": {}, "surface_audits": [], "max_depth": 0, "total_surfaces": 0}

    slices: dict = {}       # depth → AxisSlice
    surface_audits = []

    for i, unit in enumerate(units):
        depth = unit.stack_depth

        # ── Descent: add this unit to its depth slice ─────────────────────
        if depth not in slices:
            slices[depth] = AxisSlice(depth=depth, units=[])

        # Use cross_switch and residue_in as pressure proxies
        pressure_proxy = (
            0.4 if unit.cross_switch else 0.05
        ) + unit.residue_in
        slices[depth].add_unit(i, pressure_proxy)

        # ── Ascent: detect surface events from unit deltas ────────────────
        if i == 0:
            continue

        prev = units[i - 1]
        depth_delta  = unit.stack_depth - prev.stack_depth
        switch_delta = unit.switch_index - prev.switch_index

        # POP: ascending from prev.stack_depth to unit.stack_depth
        # Fire governance: surface content vs axis built at prev depth
        if depth_delta < 0:
            prev_depth = prev.stack_depth
            if prev_depth in slices and slices[prev_depth].unit_count > 0:
                slices[prev_depth].committed = True
                audit = slices[prev_depth].audit_against(pressure_proxy)
                audit["event"]    = "POP"
                audit["unit_idx"] = i
                audit["from_depth"] = prev_depth
                surface_audits.append(audit)

        # SWITCH: lateral — fire governance: current vs axis at this depth
        if switch_delta > 0:
            if depth in slices and slices[depth].unit_count > 1:
                audit = slices[depth].audit_against(pressure_proxy)
                audit["event"]    = "SWITCH"
                audit["unit_idx"] = i
                audit["from_depth"] = depth
                surface_audits.append(audit)

    return {
        "slices":         slices,
        "surface_audits": surface_audits,
        "max_depth":      max(slices.keys()) if slices else 0,
        "total_surfaces": len(surface_audits),
        "nack_count":     sum(1 for a in surface_audits if a["decision"] == "NACK"),
        "fin_count":      sum(1 for a in surface_audits if a["decision"] == "FIN"),
    }



# ══════════════════════════════════════════════════════════════════════════════
# Perspective Depth Pass — Ego-Dissolution Extension
# ══════════════════════════════════════════════════════════════════════════════
#
# The base parse_context_units() fires on explicit lexical markers (SWITCH).
# All contrastive pivots land at the same depth — lateral moves.
#
# Ego-dissolution (IEP v1.0 / v2.0) recognises that some SWITCH events are
# actually PUSH/POP pairs: you step INTO a new perspective (depth increases),
# inhabit it fully, then return to the parent frame (depth decreases).
#
# "However, Searle argues..." is not a lateral SWITCH.
# It is a PUSH: you leave the functionalist frame, enter Searle's frame.
# The parent frame (Putnam/Lewis/Armstrong) is preserved on the stack.
#
# "Nevertheless, the functionalist core survives..." is the POP:
# you ascend back to the parent frame. The stack carries what was committed.
#
# This second pass re-classifies SWITCH units as PUSH or POP based on:
#   1. Perspective entry markers  → reclassify as PUSH (depth++)
#   2. Perspective return markers → reclassify as POP  (depth--)
#   3. Anchor density per unit    → load-bearing vs. elaboration
#   4. Transition friction        → Sandhi (0.0) vs. Glottal Stop (1.0)
#
# New fields added to ContextUnit:
#   perspective_op:    "HOLD" | "PUSH" | "POP" | "SWITCH"  (reclassified)
#   anchor_density:    float [0,1]  fraction of sentences that are load-bearing
#   friction:          float [0,1]  transition cost from prior unit
#   omega_direction:   float [-1,1] +1=moving away from axis, -1=returning
# ══════════════════════════════════════════════════════════════════════════════

import re
from dataclasses import dataclass
from typing import List, Optional


# ── Perspective entry markers (PUSH candidates) ───────────────────────────────
# These open a nested perspective: a named position, a counterargument,
# a thought experiment. The prior frame goes onto the stack.

PERSPECTIVE_PUSH = re.compile(
    r'(?:^|(?<=\.)\s+|\n)\s*'
    r'(however[,\s]|yet[,\s]|but\s+(?!because)'
    r'|by\s+contrast[,\s]|in\s+contrast[,\s]'
    r'|on\s+the\s+other\s+hand'
    r'|consider[,\s]|suppose[,\s]|imagine[,\s]'
    r'|(?:[\w]+\s+){0,3}(?:argues?|claims?|objects?|challenges?|'
    r'proposes?|suggests?|holds?\s+that|maintains?\s+that|'
    r'insists?\s+that|contends?\s+that)\b)',
    re.IGNORECASE | re.MULTILINE
)

# ── Perspective return markers (POP candidates) ────────────────────────────────
# These signal ascent back to the prior frame.

PERSPECTIVE_POP = re.compile(
    r'(?:^|(?<=\.)\s+|\n)\s*'
    r'(nevertheless[,\s]|nonetheless[,\s]'
    r'|despite\s+(?:this|these)[,\s]'
    r'|even\s+so[,\s]|that\s+said[,\s]'
    r'|in\s+any\s+(?:case|event)[,\s]'
    r'|the\s+(?:honest|correct|right)\s+(?:position|answer|view)'
    r'|returning\s+to|to\s+return\s+to'
    r'|the\s+(?:core|central|fundamental)\s+(?:claim|thesis|point|insight)\s+'
    r'(?:remains?|survives?|holds?)'
    r'|in\s+(?:summary|conclusion|sum)[,\s])',
    re.IGNORECASE | re.MULTILINE
)

# ── Anchor patterns — load-bearing sentences ──────────────────────────────────
# Named entities + specific claims = anchors.
# Elaboration, hedging, atmosphere = voids.

ANCHOR_SIGNALS = re.compile(
    r'('
    # Specific named entities (proper nouns, years, named theories)
    r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'      # proper names
    r'|\b(?:19|20)\d{2}\b'                  # years
    r'|\b(?:iff?|iff|i\.e\.|e\.g\.)\b'     # formal connectives
    # Causal / definitional claims
    r'|(?:is\s+defined\s+as|is\s+identical\s+to|'
    r'equals?|is\s+the\s+(?:claim|thesis|view)\s+that|'
    r'establishes?\s+that|proves?\s+that|shows?\s+that|'
    r'entails?\s+that|follows?\s+from)'
    # Mathematical / quantified claims
    r'|\b\d+(?:\.\d+)?(?:\s*%|\s*tokens?|\s*joules?)\b'
    r')',
    re.IGNORECASE
)

VOID_SIGNALS = re.compile(
    r'('
    r'\b(?:might|may|perhaps|possibly|arguably|'
    r'in\s+some\s+sense|to\s+some\s+extent|'
    r'it\s+(?:seems?|appears?|could\s+be)|'
    r'one\s+(?:might|could|may)\s+(?:say|argue|think)|'
    r'this\s+is\s+(?:not|merely|just)|'
    r'broadly\s+speaking|in\s+general)\b'
    r')',
    re.IGNORECASE
)


def anchor_density(text: str) -> float:
    """
    Fraction of the unit that is load-bearing (anchor) vs. atmospheric (void).

    Anchors: named entities, specific dates, quantified claims, definitions.
    Voids:   hedging, elaboration, qualification, rhetorical questions.

    High anchor density → axis material (protect under budget pressure).
    Low anchor density  → elaboration (evict first within same switch tier).
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0

    anchor_count = 0
    for sent in sentences:
        anchors = len(ANCHOR_SIGNALS.findall(sent))
        voids   = len(VOID_SIGNALS.findall(sent))
        # A sentence is load-bearing if anchors dominate
        if anchors > voids and anchors >= 1:
            anchor_count += 1

    return anchor_count / len(sentences)


def transition_friction(unit_a: 'ContextUnit', unit_b: 'ContextUnit') -> float:
    """
    Transition cost between two adjacent units.
    Mirrors IEP v1.0 Friction Coefficient:

      F ≈ 0.0  Sandhi     — same conceptual field, continuation
      F ≈ 0.5  Soft switch — subject drift, register shift
      F ≈ 1.0  Glottal stop — hard break, new perspective enters

    Measured by:
      - Named entity overlap (low overlap → high friction)
      - Perspective marker at unit_b start (PUSH/POP → elevated friction)
      - Anchor density delta (large shift → higher friction)
    """
    def extract_proper_nouns(text: str):
        return set(re.findall(r'\b[A-Z][a-z]{2,}\b', text))

    nouns_a = extract_proper_nouns(unit_a.text)
    nouns_b = extract_proper_nouns(unit_b.text)

    # Named entity overlap [0,1]
    union        = nouns_a | nouns_b
    intersection = nouns_a & nouns_b
    ne_overlap   = len(intersection) / max(len(union), 1)

    # Perspective entry at b start → high friction
    persp_friction = 0.0
    first_100 = unit_b.text[:100]
    if PERSPECTIVE_PUSH.match(first_100):
        persp_friction = 0.5
    elif PERSPECTIVE_POP.match(first_100):
        persp_friction = 0.3   # return is smoother than entry

    # Anchor density delta
    ad_a     = anchor_density(unit_a.text)
    ad_b     = anchor_density(unit_b.text)
    ad_delta = abs(ad_a - ad_b) * 0.3

    # Combine: low overlap + perspective entry = high friction
    raw = (1.0 - ne_overlap) * 0.5 + persp_friction + ad_delta
    return min(1.0, raw)


def omega_direction(unit: 'ContextUnit',
                    axis_unit: 'ContextUnit') -> float:
    """
    Directional Ω: is this unit moving toward or away from the axis?

      +1.0  moving away  (counterargument, new perspective, high friction)
      -1.0  returning    (synthesis, return marker, axis vocabulary)
       0.0  neutral      (elaboration within same frame)

    This is the key CDP signal that scalar Ω cannot provide.
    """
    text_low = unit.text.lower()

    # Return markers → negative direction (moving toward axis)
    if PERSPECTIVE_POP.search(unit.text[:150]):
        return -0.7

    # Entry markers → positive direction (moving away from axis)
    if PERSPECTIVE_PUSH.search(unit.text[:150]):
        return +0.7

    # Anchor overlap with axis unit
    axis_nouns = set(re.findall(r'\b[A-Z][a-z]{2,}\b', axis_unit.text))
    unit_nouns = set(re.findall(r'\b[A-Z][a-z]{2,}\b', unit.text))
    overlap    = len(axis_nouns & unit_nouns) / max(len(axis_nouns), 1)

    # High overlap with axis → returning; low overlap → diverging
    if overlap > 0.4:
        return -0.3   # axis-adjacent
    elif overlap < 0.1:
        return +0.3   # axis-distant
    return 0.0


@dataclass
class PerspectiveUnit:
    """
    ContextUnit enriched with ego-dissolution perspective analysis.
    Wraps the original unit without modifying it.
    """
    unit:             'ContextUnit'
    perspective_op:   str    # HOLD | PUSH | POP | SWITCH (reclassified)
    anchor_density:   float  # [0,1] load-bearing fraction
    friction:         float  # [0,1] transition cost from prior unit
    omega_direction:  float  # [-1,+1] toward/away from axis
    perspective_depth: int   # depth after perspective reclassification

    # Convenience pass-throughs
    @property
    def idx(self):          return self.unit.idx
    @property
    def text(self):         return self.unit.text
    @property
    def switch_index(self): return self.unit.switch_index
    @property
    def stack_depth(self):  return self.unit.stack_depth
    @property
    def cross_switch(self): return self.unit.cross_switch


def perspective_depth_pass(units: List['ContextUnit']) -> List[PerspectiveUnit]:
    """
    Second pass: ego-dissolution reclassification.

    Takes the flat SWITCH list from parse_context_units() and re-classifies
    each unit as PUSH (perspective entry), POP (perspective return), or HOLD.

    The stack is simulated: perspective_depth tracks how deeply we have
    descended from the axis frame. A POP decrements it back toward 0.

    Returns a list of PerspectiveUnit — enriched, not replacing, the original.
    The original ContextUnit list is unchanged.

    Usage:
        units  = parse_context_units(text)
        p_units = perspective_depth_pass(units)

        # For eviction priority: use perspective_depth + omega_direction
        # instead of switch_index alone.
        # POP units with low perspective_depth are axis-return units —
        # they should be kept even under budget pressure.
    """
    if not units:
        return []

    axis_unit        = units[0]
    p_depth          = 0
    result: List[PerspectiveUnit] = []

    for i, unit in enumerate(units):
        # Compute anchor density for this unit
        ad = anchor_density(unit.text)

        # Compute transition friction from prior unit
        friction = 0.0
        if i > 0:
            friction = transition_friction(units[i - 1], unit)

        # Compute omega direction relative to axis
        od = omega_direction(unit, axis_unit)

        # Reclassify perspective operation
        if i == 0:
            op = "HOLD"   # axis unit — always HOLD
        else:
            first_150 = unit.text[:150]
            is_push   = bool(PERSPECTIVE_PUSH.search(first_150))
            is_pop    = bool(PERSPECTIVE_POP.search(first_150))

            # Disambiguation: "Nevertheless" is usually POP.
            # "However" is usually PUSH. Both are SWITCH in base parse.
            # If both markers present, trust direction signal.
            if is_pop and not is_push:
                op = "POP"
                p_depth = max(0, p_depth - 1)
            elif is_push and not is_pop:
                op = "PUSH"
                p_depth += 1
            elif is_push and is_pop:
                # Ambiguous — use omega direction
                op = "POP" if od < 0 else "PUSH"
                p_depth = max(0, p_depth - 1) if od < 0 else p_depth + 1
            else:
                # No explicit perspective marker
                # Use friction + omega direction to decide
                if friction > 0.65 and od > 0.2:
                    op = "PUSH"
                    p_depth += 1
                elif od < -0.2 and p_depth > 0:
                    op = "POP"
                    p_depth = max(0, p_depth - 1)
                else:
                    op = "HOLD"

        result.append(PerspectiveUnit(
            unit              = unit,
            perspective_op    = op,
            anchor_density    = ad,
            friction          = friction,
            omega_direction   = od,
            perspective_depth = p_depth,
        ))

    return result


def perspective_eviction_priority(p_unit: PerspectiveUnit) -> tuple:
    """
    Eviction priority using perspective depth + anchor density.
    Lower tuple = higher priority (keep first).

    Replaces the simple switch_index > 0 heuristic in Condition C / D.

    Tier 0: axis unit (idx=0) — always keep
    Tier 1: HOLD units at perspective_depth=0 — axis elaboration
    Tier 2: POP units (returning to axis) — axis-convergent
    Tier 3: HOLD units at perspective_depth>0 — nested elaboration
    Tier 4: PUSH units — perspective descent
    Tier 5: PUSH units with low anchor density — atmospheric descents

    Within each tier: higher anchor_density = higher priority.
    """
    if p_unit.idx == 0:
        return (0, 0.0)

    op    = p_unit.perspective_op
    depth = p_unit.perspective_depth
    ad    = p_unit.anchor_density
    od    = p_unit.omega_direction

    if op == "HOLD" and depth == 0:
        return (1, -ad)   # axis elaboration, rank by anchor density
    if op == "POP":
        return (2, -ad)   # return to axis — keep even under pressure
    if op == "HOLD" and depth > 0:
        return (3, depth - ad)
    if op == "PUSH" and ad > 0.4:
        return (4, -ad)   # load-bearing push
    if op == "PUSH":
        return (5, -ad)   # atmospheric push — evict first

    return (3, -ad)       # fallback
