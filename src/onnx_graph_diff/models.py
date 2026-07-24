"""Plain dataclasses shared across the matching, proof, and reporting layers.

These types are the vocabulary the whole pipeline speaks in:

* :class:`Region` / :class:`AnchorCandidate` -- a closed set of region nodes and
  a proposed rejoin anchor pair produced by a matching route.
* :class:`EquivalenceResult` -- the verdict returned by the equivalence checker.
* :class:`MatchEvent` -- one step of the alignment (an exact pair, a validated
  difference region, or an unresolved region), the unit the report renders.
* :class:`LLMMatchProposal` / :class:`AgenticMatchProposal` -- structured LLM
  outputs the assisted routes parse before re-validating them.
* :class:`LLMCallStats` -- per-call token/latency accounting.
* :class:`ComparisonResult` -- the top-level result, JSON-serializable via
  :meth:`ComparisonResult.to_dict`.

Everything here is a dataclass with no behaviour beyond serialization, so the
layers stay decoupled and results round-trip cleanly to JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ProofStatus = Literal["proven", "structural", "hypothesis", "rejected"]
ComparisonMethod = Literal["fingerprint", "llm", "llm_agent"]


@dataclass(frozen=True)
class Region:
    node_ids: tuple[str, ...]
    entry_values: tuple[str, ...]
    exit_values: tuple[str, ...]


@dataclass(frozen=True)
class AnchorCandidate:
    left_anchor: str
    right_anchor: str
    left_region: Region
    right_region: Region
    exit_mapping: tuple[tuple[str, str], ...]
    score: float
    evidence: tuple[str, ...] = ()


@dataclass
class EquivalenceResult:
    status: ProofStatus
    rule_id: str | None = None
    details: list[str] = field(default_factory=list)


@dataclass
class MatchEvent:
    kind: Literal["exact", "region", "unresolved"]
    left_nodes: list[str]
    right_nodes: list[str]
    left_anchor: str | None = None
    right_anchor: str | None = None
    value_mapping: list[tuple[str, str]] = field(default_factory=list)
    proof: EquivalenceResult | None = None
    source: Literal["fingerprint", "llm", "llm_agent", "deterministic"] = "deterministic"
    confidence: float = 1.0
    explanation: str | None = None


@dataclass
class LLMCallStats:
    model: str
    context_strategy: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float | None = None
    thinking: bool = False
    seed: int | None = None
    parse_ok: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    method: ComparisonMethod
    completed: bool
    events: list[MatchEvent]
    value_mapping: dict[str, str]
    unmatched_left: list[str]
    unmatched_right: list[str]
    llm_calls: list[LLMCallStats] = field(default_factory=list)
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMMatchProposal:
    left_anchor: str | None
    right_anchor: str | None
    left_region_nodes: list[str]
    right_region_nodes: list[str]
    exit_mapping: list[tuple[str, str]]
    rule_id: str | None
    equivalence_claim: Literal["exact", "algebraic", "approximate", "unknown"]
    confidence: float
    conditions: list[str]
    explanation: str


@dataclass
class AgenticMatchProposal:
    action: Literal["match", "expand", "unresolved"]
    left_anchor: str | None = None
    right_anchor: str | None = None
    left_region_nodes: list[str] = field(default_factory=list)
    right_region_nodes: list[str] = field(default_factory=list)
    exit_mapping: list[tuple[str, str]] = field(default_factory=list)
    rule_id: str | None = None
    equivalence_claim: Literal["exact", "algebraic", "approximate", "unknown"] = "unknown"
    confidence: float = 0.0
    conditions: list[str] = field(default_factory=list)
    explanation: str = ""
    expand_hops: int = 2
    expand_nodes: int = 8
    same_op_pairs_seen: list[tuple[str, str]] = field(default_factory=list)
