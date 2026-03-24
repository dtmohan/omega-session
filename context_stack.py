"""
context_stack.py — Ingestion Layer
Segments conversation turns by argument AST.
Minimal Topology primitive — no dependencies.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List

# Switch markers that trigger PUSH/POP/SWITCH/RESET
SWITCH_MARKERS = [
    r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\byet\b',
    r'\btherefore\b', r'\bthus\b', r'\bhence\b', r'\bconsequently\b',
    r'\bif\b', r'\bunless\b', r'\bwhereas\b', r'\bdespite\b',
    r'\bnevertheless\b', r'\bnonetheless\b', r'^#{1,3}\s', r'^---',
]
SWITCH_PATTERN = re.compile('|'.join(SWITCH_MARKERS), re.IGNORECASE | re.MULTILINE)

QUALIFIER_OPEN  = re.compile(r'\b(assuming|given that|provided that|in the case that)\b', re.I)
QUALIFIER_CLOSE = re.compile(r'\b(in conclusion|to summarize|in summary|therefore)\b', re.I)


@dataclass
class ContextUnit:
    text: str
    role: str                    # 'user' | 'assistant'
    turn_index: int
    switch_index: int            # position in argument tree
    stack_depth: int             # nesting depth
    cross_switch: bool           # opens with causal bridge → PROVISIONAL
    char_len: int = 0
    token_estimate: int = 0
    unit_hash: str = ""

    def __post_init__(self):
        self.char_len = len(self.text)
        self.token_estimate = max(1, self.char_len // 4)
        self.unit_hash = hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class ComplexityCounters:
    stack_depth: int
    context_switches: int
    estimated_window: int
    unit_count: int
    total_tokens: int


def parse_context_units(turns: List[dict]) -> List[ContextUnit]:
    """
    Parse conversation turns into ContextUnits.
    Each turn is {'role': str, 'content': str, 'turn_index': int}.
    Sub-segments within a turn by switch markers.
    """
    units = []
    switch_index = 0
    stack_depth = 0

    for turn in turns:
        text = turn['content'].strip()
        role = turn['role']
        tidx = turn['turn_index']

        # Split turn into sub-segments at switch markers
        segments = _split_at_switches(text)

        for i, seg in enumerate(segments):
            if not seg.strip():
                continue

            cross_switch = (i > 0) or bool(SWITCH_PATTERN.match(seg.strip()))

            # Track depth
            if QUALIFIER_OPEN.search(seg):
                stack_depth += 1
            if QUALIFIER_CLOSE.search(seg):
                stack_depth = max(0, stack_depth - 1)

            unit = ContextUnit(
                text=seg.strip(),
                role=role,
                turn_index=tidx,
                switch_index=switch_index,
                stack_depth=stack_depth,
                cross_switch=cross_switch,
            )
            units.append(unit)
            switch_index += 1

    return units


def complexity_counters(units: List[ContextUnit]) -> ComplexityCounters:
    if not units:
        return ComplexityCounters(0, 0, 500, 0, 0)

    max_depth    = max(u.stack_depth for u in units)
    switches     = sum(1 for u in units if u.cross_switch)
    total_tokens = sum(u.token_estimate for u in units)

    # IEP window formula
    estimated_window = 500 + (max_depth * 200) + (switches * 100)

    return ComplexityCounters(
        stack_depth=max_depth,
        context_switches=switches,
        estimated_window=estimated_window,
        unit_count=len(units),
        total_tokens=total_tokens,
    )


def _split_at_switches(text: str) -> List[str]:
    """Split text into segments at switch boundaries."""
    positions = [0]
    for m in SWITCH_PATTERN.finditer(text):
        if m.start() > 0:
            positions.append(m.start())
    positions.append(len(text))

    segments = []
    for i in range(len(positions) - 1):
        seg = text[positions[i]:positions[i+1]]
        if seg.strip():
            segments.append(seg)
    return segments if segments else [text]
