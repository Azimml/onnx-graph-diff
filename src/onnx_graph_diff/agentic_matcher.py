from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .context import ContextConfig, GraphContextBuilder
from .equivalence import StaticEquivalenceValidator
from .explanation import build_explanation_summary
from .fingerprint import FingerprintIndex
from .ir import GraphIR
from .llm_client import AGENTIC_SYSTEM_PROMPTS, AgenticLLMInvocation, LLMClient
from .matcher import FingerprintMatcher, MatcherConfig
from .models import ComparisonResult, MatchEvent


class AgenticMatchClient(Protocol):
    def propose_agentic_match(
        self,
        context: dict[str, Any],
        *,
        context_strategy: str,
        prompt_variant: str = "agent_v13",
        feedback: str | None = None,
    ) -> AgenticLLMInvocation: ...

    def explain(self, summary: dict[str, Any]) -> tuple[str, object]: ...


@dataclass
class AgenticMatcherConfig:
    context_strategy: str = "bounded"
    prompt_variant: str = "agent_v13"
    initial_hops: int = 2
    initial_nodes_per_graph: int = 8
    max_hops: int = 8
    max_nodes_per_graph: int = 32
    max_chars: int = 32_000
    max_expansions: int = 3
    max_retries_per_mismatch: int = 2
    max_calls_per_mismatch: int = 8
    allow_structural_unknown: bool = True
    explain: bool = False
    capture_transcripts: bool = False


class LLMAgenticMatcher:
    """LLM-led anchor search with deterministic graph-state and proof checks."""

    def __init__(
        self,
        client: AgenticMatchClient | LLMClient,
        config: AgenticMatcherConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or AgenticMatcherConfig()
        self.mechanics = FingerprintMatcher(
            MatcherConfig(
                max_hops=self.config.max_hops,
                max_window_nodes=self.config.max_nodes_per_graph * 2,
                allow_structural_unknown=self.config.allow_structural_unknown,
            )
        )
        self.validator = StaticEquivalenceValidator()

    def compare(self, left: GraphIR, right: GraphIR) -> ComparisonResult:
        left_fp = FingerprintIndex(left)
        right_fp = FingerprintIndex(right)
        value_mapping = self.mechanics._initial_input_mapping(left, right)
        matched_left: set[str] = set()
        matched_right: set[str] = set()
        events: list[MatchEvent] = []
        llm_calls = []
        total_expansions = 0

        while len(matched_left) < len(left.nodes) or len(matched_right) < len(right.nodes):
            ready_left = self.mechanics._ready(left, matched_left)
            ready_right = self.mechanics._ready(right, matched_right)
            exact_pairs = self.mechanics._exact_pairs(
                left,
                right,
                ready_left,
                ready_right,
                value_mapping,
                left_fp,
                right_fp,
            )
            if exact_pairs:
                for left_id, right_id in exact_pairs:
                    self.mechanics._commit_exact(
                        left,
                        right,
                        left_id,
                        right_id,
                        value_mapping,
                        matched_left,
                        matched_right,
                    )
                    events.append(
                        MatchEvent(
                            kind="exact",
                            left_nodes=[left_id],
                            right_nodes=[right_id],
                            value_mapping=list(
                                zip(left.nodes[left_id].outputs, right.nodes[right_id].outputs, strict=False)
                            ),
                            source="deterministic",
                            explanation=(
                                "Exact match accepted without an LLM call because the local node signatures match, "
                                "all dynamic inputs are already paired by value_mapping, constant roles and hashes "
                                "match, and output dtype/shape descriptors are compatible."
                            ),
                        )
                    )
                continue

            accepted = False
            feedback: str | None = None
            rejected_anchor_pairs: set[tuple[str, str]] = set()
            rejected_regions = []
            rejected_decisions: list[dict[str, Any]] = []
            expansion_history: list[dict[str, Any]] = []
            expansion_round = 0
            parse_failures = 0
            hops = min(self.config.initial_hops, self.config.max_hops)
            nodes_per_graph = min(self.config.initial_nodes_per_graph, self.config.max_nodes_per_graph)

            for call_index in range(self.config.max_calls_per_mismatch):
                can_expand = expansion_round < self.config.max_expansions and (
                    hops < self.config.max_hops or nodes_per_graph < self.config.max_nodes_per_graph
                )
                context = self._build_context(
                    left,
                    right,
                    ready_left,
                    ready_right,
                    matched_left,
                    matched_right,
                    value_mapping,
                    hops=hops,
                    nodes_per_graph=nodes_per_graph,
                    expansion_round=expansion_round,
                    can_expand=can_expand,
                    expansion_history=expansion_history,
                    rejected_decisions=rejected_decisions,
                )
                can_expand = bool(context["agent_control"]["can_expand"])
                invocation = self.client.propose_agentic_match(
                    context,
                    context_strategy=self.config.context_strategy,
                    prompt_variant=self.config.prompt_variant,
                    feedback=feedback,
                )
                invocation.stats.metadata.update(
                    {
                        "call_index": call_index,
                        "accepted": False,
                        "expansion_round": expansion_round,
                        "window_hops": hops,
                        "window_nodes_per_graph": nodes_per_graph,
                        "can_expand": can_expand,
                        "context_chars": len(json.dumps(context, ensure_ascii=False, default=str)),
                        "left_visible_nodes": len(context["left"]["visible_node_ids"]),
                        "right_visible_nodes": len(context["right"]["visible_node_ids"]),
                        "left_visible_anchor_ids": context["left"]["visible_anchor_ids"],
                        "right_visible_anchor_ids": context["right"]["visible_anchor_ids"],
                        "coarse_candidates_supplied": False,
                    }
                )
                if self.config.capture_transcripts:
                    invocation.stats.metadata["raw_response"] = invocation.raw_content
                    invocation.stats.metadata["system_prompt"] = AGENTIC_SYSTEM_PROMPTS.get(self.config.prompt_variant)
                    invocation.stats.metadata["user_context"] = context
                    invocation.stats.metadata["feedback"] = feedback
                llm_calls.append(invocation.stats)
                proposal = invocation.proposal
                if proposal is None:
                    parse_failures += 1
                    feedback = f"The previous response was not valid JSON: {invocation.parse_error}"
                    if parse_failures > self.config.max_retries_per_mismatch:
                        break
                    continue

                invocation.stats.metadata.update(
                    {
                        "action": proposal.action,
                        "claimed_rule": proposal.rule_id,
                        "proposed_left_region": proposal.left_region_nodes,
                        "proposed_right_region": proposal.right_region_nodes,
                        "proposed_exit_mapping": proposal.exit_mapping,
                        "same_op_pairs_seen": [list(pair) for pair in proposal.same_op_pairs_seen],
                    }
                )
                if proposal.action == "expand":
                    if not can_expand:
                        feedback = "The graph window is already at its expansion limit. Return match or unresolved."
                        continue
                    new_hops = min(self.config.max_hops, hops + proposal.expand_hops)
                    new_nodes = min(self.config.max_nodes_per_graph, nodes_per_graph + proposal.expand_nodes)
                    if new_hops == hops and new_nodes == nodes_per_graph:
                        feedback = "The requested expansion did not reveal more graph. Return match or unresolved."
                        continue
                    expansion_history.append(
                        {
                            "round": expansion_round,
                            "from_hops": hops,
                            "to_hops": new_hops,
                            "from_nodes": nodes_per_graph,
                            "to_nodes": new_nodes,
                            "reason": proposal.explanation,
                        }
                    )
                    invocation.stats.metadata["expanded"] = True
                    hops = new_hops
                    nodes_per_graph = new_nodes
                    expansion_round += 1
                    total_expansions += 1
                    feedback = (
                        f"Expanded to max_hops={hops}, max_nodes_per_graph={nodes_per_graph}. "
                        "Inspect the newly visible nodes and choose match, expand, or unresolved."
                    )
                    continue

                if proposal.action == "unresolved":
                    feedback = proposal.explanation or "The LLM found no defensible rejoin point."
                    invocation.stats.metadata["accepted_action"] = "unresolved"
                    break

                if proposal.left_anchor is None or proposal.right_anchor is None:
                    feedback = "action=match requires both anchor IDs. Request expansion instead of guessing."
                    continue
                anchor_pair = (proposal.left_anchor, proposal.right_anchor)
                visible_left = set(context["left"]["visible_anchor_ids"])
                visible_right = set(context["right"]["visible_anchor_ids"])
                if proposal.left_anchor not in visible_left or proposal.right_anchor not in visible_right:
                    feedback = (
                        "One or both anchors are outside the visible window. "
                        "Use action=expand if more graph is needed; never invent IDs."
                    )
                    rejected_decisions.append({"anchors": list(anchor_pair), "reason": feedback})
                    continue
                if anchor_pair in rejected_anchor_pairs:
                    feedback = "That anchor pair was already rejected. Choose another visible pair or expand."
                    continue
                if (
                    not proposal.left_anchor.startswith("__graph_output__:")
                    and anchor_pair not in proposal.same_op_pairs_seen
                ):
                    feedback = (
                        "The selected anchor pair was not included in same_op_pairs_seen. "
                        "Report your visible-node scan as [[left_id,right_id], ...] and include the selected pair."
                    )
                    continue

                candidate = self.mechanics.build_candidate_from_anchors(
                    left,
                    right,
                    proposal.left_anchor,
                    proposal.right_anchor,
                    matched_left,
                    matched_right,
                    value_mapping,
                )
                if candidate is None:
                    feedback = self._diagnose_anchor_pair(
                        left,
                        right,
                        proposal.left_anchor,
                        proposal.right_anchor,
                    )
                    rejected_decisions.append({"anchors": list(anchor_pair), "reason": feedback})
                    continue

                proof = self.validator.validate(left, right, candidate, value_mapping)
                invocation.stats.metadata.update(
                    {
                        "validated_rule": proof.rule_id,
                        "proof_status": proof.status,
                        "anchor_pair": [proposal.left_anchor, proposal.right_anchor],
                        "reconstructed_left_region": list(candidate.left_region.node_ids),
                        "reconstructed_right_region": list(candidate.right_region.node_ids),
                        "reconstructed_exit_mapping": list(candidate.exit_mapping),
                    }
                )
                if proof.status == "rejected":
                    rejected_anchor_pairs.add(anchor_pair)
                    rejected_regions.append(candidate)
                    feedback = "; ".join(proof.details) + ". Choose another visible pair or request expansion."
                    rejected_decisions.append({"anchors": list(anchor_pair), "reason": feedback})
                    continue
                if proof.status == "structural" and not self.config.allow_structural_unknown:
                    rejected_anchor_pairs.add(anchor_pair)
                    feedback = "The selected region has no supported static proof certificate."
                    rejected_decisions.append({"anchors": list(anchor_pair), "reason": feedback})
                    continue
                if proof.status == "structural" and any(
                    set(rejected.left_region.node_ids).issubset(candidate.left_region.node_ids)
                    and set(rejected.right_region.node_ids).issubset(candidate.right_region.node_ids)
                    for rejected in rejected_regions
                ):
                    rejected_anchor_pairs.add(anchor_pair)
                    feedback = "This region contains a smaller subregion already rejected by a known certificate."
                    rejected_decisions.append({"anchors": list(anchor_pair), "reason": feedback})
                    continue
                if proof.status == "structural" and "same full downstream fingerprint" not in candidate.evidence:
                    rejected_anchor_pairs.add(anchor_pair)
                    feedback = (
                        "The unknown transformation lacks a stable downstream rejoin. "
                        "Choose a later unchanged pair or request expansion."
                    )
                    rejected_decisions.append({"anchors": list(anchor_pair), "reason": feedback})
                    continue

                self.mechanics._commit_region(candidate, value_mapping, matched_left, matched_right)
                invocation.stats.metadata["accepted"] = True
                events.append(
                    MatchEvent(
                        kind="region",
                        left_nodes=list(candidate.left_region.node_ids),
                        right_nodes=list(candidate.right_region.node_ids),
                        left_anchor=candidate.left_anchor,
                        right_anchor=candidate.right_anchor,
                        value_mapping=list(candidate.exit_mapping),
                        proof=proof,
                        source="llm_agent",
                        confidence=proposal.confidence,
                        explanation=proposal.explanation,
                    )
                )
                accepted = True
                break

            if accepted:
                continue
            events.append(
                MatchEvent(
                    kind="unresolved",
                    left_nodes=ready_left,
                    right_nodes=ready_right,
                    source="llm_agent",
                    confidence=0.0,
                    explanation=feedback,
                )
            )
            break

        unmatched_left = [node_id for node_id in left.node_order if node_id not in matched_left]
        unmatched_right = [node_id for node_id in right.node_order if node_id not in matched_right]
        result = ComparisonResult(
            method="llm_agent",
            completed=not unmatched_left and not unmatched_right,
            events=events,
            value_mapping=value_mapping,
            unmatched_left=unmatched_left,
            unmatched_right=unmatched_right,
            llm_calls=llm_calls,
            metadata={
                "left_graph": left.name,
                "right_graph": right.name,
                "context_strategy": self.config.context_strategy,
                "prompt_variant": self.config.prompt_variant,
                "initial_hops": self.config.initial_hops,
                "initial_nodes_per_graph": self.config.initial_nodes_per_graph,
                "max_hops": self.config.max_hops,
                "max_nodes_per_graph": self.config.max_nodes_per_graph,
                "max_expansions": self.config.max_expansions,
                "total_expansions": total_expansions,
                "coarse_candidates_supplied": False,
            },
        )
        if self.config.explain:
            explanation, stats = self.client.explain(build_explanation_summary(left, right, result))
            result.explanation = explanation
            result.llm_calls.append(stats)  # type: ignore[arg-type]
        return result

    def _build_context(
        self,
        left: GraphIR,
        right: GraphIR,
        ready_left: list[str],
        ready_right: list[str],
        matched_left: set[str],
        matched_right: set[str],
        value_mapping: dict[str, str],
        *,
        hops: int,
        nodes_per_graph: int,
        expansion_round: int,
        can_expand: bool,
        expansion_history: list[dict[str, Any]],
        rejected_decisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        builder = GraphContextBuilder(
            ContextConfig(
                strategy=self.config.context_strategy,  # type: ignore[arg-type]
                max_hops=hops,
                max_nodes_per_graph=nodes_per_graph,
                max_chars=self.config.max_chars,
                include_coarse_anchor_pairs=False,
            )
        )
        context = builder.build(
            left,
            right,
            ready_left,
            ready_right,
            matched_left,
            matched_right,
            value_mapping,
        )
        self._augment_side(left, context["left"], set(ready_left), matched_left)
        self._augment_side(right, context["right"], set(ready_right), matched_right)
        hidden_downstream = self._has_hidden_downstream(context["left"]) or self._has_hidden_downstream(
            context["right"]
        )
        effective_can_expand = can_expand and hidden_downstream
        context["agent_control"] = {
            "route": "llm_agent",
            "candidate_pairs_are_not_supplied": True,
            "allowed_actions": ["match", "expand", "unresolved"],
            "expansion_round": expansion_round,
            "current_max_hops": hops,
            "current_max_nodes_per_graph": nodes_per_graph,
            "maximum_hops": self.config.max_hops,
            "maximum_nodes_per_graph": self.config.max_nodes_per_graph,
            "can_expand": effective_can_expand,
            "hidden_downstream_detected": hidden_downstream,
            "expansion_history": expansion_history,
            "rejected_decisions": rejected_decisions[-6:],
            "hard_anchor_rules": [
                "Anchor domain/op, attributes, arity, constant roles, and outputs must match.",
                "All transformation nodes must be before the anchors; anchors are never primitive/fused region nodes.",
                "required_parent_nodes remain visible as history but are already matched and cannot be selected as anchors.",
                "Use matching virtual graph-output anchors for terminal transformations with no later unchanged node pair.",
            ],
            "match_instruction": (
                "Choose the earliest visible unchanged anchor pair after the mismatch. "
                "The program will reconstruct and validate the region from your anchors."
            ),
            "expand_instruction": (
                "Return action=expand when the likely rejoin or a required dependency is outside the visible window."
            ),
        }
        return context

    @staticmethod
    def _has_hidden_downstream(side: dict[str, Any]) -> bool:
        if side["window_boundary_node_ids"]:
            return True
        return any(not output["producer_visible"] for output in side["graph_outputs"])

    @staticmethod
    def _augment_side(
        graph: GraphIR,
        side: dict[str, Any],
        frontier: set[str],
        matched: set[str],
    ) -> None:
        visible = set(side["visible_node_ids"])
        distances = LLMAgenticMatcher._frontier_distances(graph, frontier, visible)
        for node in side["nodes"]:
            node["frontier_distance"] = distances.get(node["id"])
        boundary_nodes = [
            node_id
            for node_id in side["visible_node_ids"]
            if any(child not in visible for child in graph.children(node_id))
        ]
        graph_outputs = []
        visible_anchor_ids = [node_id for node_id in side["visible_node_ids"] if node_id not in matched]
        for index, value_name in enumerate(graph.graph_outputs):
            producer = graph.values[value_name].producer
            producer_visible = producer is None or producer in visible
            virtual_anchor = f"__graph_output__:{index}"
            if producer_visible:
                visible_anchor_ids.append(virtual_anchor)
            graph_outputs.append(
                {
                    "virtual_anchor": virtual_anchor,
                    "value": graph.values[value_name].compact(),
                    "producer": producer,
                    "producer_visible": producer_visible,
                }
            )
        side["window_boundary_node_ids"] = boundary_nodes
        side["graph_outputs"] = graph_outputs
        side["visible_matched_history_ids"] = [node_id for node_id in side["visible_node_ids"] if node_id in matched]
        side["visible_anchor_ids"] = visible_anchor_ids

    @staticmethod
    def _frontier_distances(graph: GraphIR, frontier: set[str], visible: set[str]) -> dict[str, int]:
        queue = [(node_id, 0) for node_id in frontier]
        distances: dict[str, int] = {}
        while queue:
            node_id, distance = queue.pop(0)
            if node_id in distances and distances[node_id] <= distance:
                continue
            distances[node_id] = distance
            queue.extend((child, distance + 1) for child in graph.children(node_id) if child in visible)
        return distances

    @staticmethod
    def _diagnose_anchor_pair(left: GraphIR, right: GraphIR, left_anchor: str, right_anchor: str) -> str:
        output_prefix = "__graph_output__:"
        if left_anchor.startswith(output_prefix) or right_anchor.startswith(output_prefix):
            return (
                "Virtual output anchors must use the same valid __graph_output__:i ID on both sides, "
                "with a compatible closed output region."
            )
        if left_anchor not in left.nodes or right_anchor not in right.nodes:
            return "One or both anchor IDs do not exist in the graphs."
        left_node = left.nodes[left_anchor]
        right_node = right.nodes[right_anchor]
        if (left_node.domain, left_node.op_type) != (right_node.domain, right_node.op_type):
            return (
                "The selected anchors have different domain/op types and cannot resume exact matching. "
                "Primitive/fused nodes such as MatMul/Add versus Gemm belong inside the region; choose the next "
                "same-operation pair, or matching virtual graph outputs for a terminal transformation."
            )
        if (len(left_node.inputs), len(left_node.outputs)) != (len(right_node.inputs), len(right_node.outputs)):
            return "The selected anchors have different input/output arity."
        if left.local_signature(left_anchor) != right.local_signature(right_anchor):
            return (
                "The selected anchors differ in attributes, output types/shapes, or constant input roles. "
                "The pair may be inside the transformation; choose a later same-operation pair after all changed nodes."
            )
        return (
            "The anchors look locally compatible, but their backward region does not form closed mapped entry cuts. "
            "Choose a later pair or request expansion."
        )
