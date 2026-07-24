from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from .context import ContextConfig, GraphContextBuilder
from .equivalence import StaticEquivalenceValidator
from .explanation import build_explanation_summary
from .fingerprint import FingerprintIndex
from .ir import GraphIR
from .llm_client import LLMClient, LLMInvocation
from .matcher import FingerprintMatcher, MatcherConfig
from .models import ComparisonResult, MatchEvent


class MatchClient(Protocol):
    def propose_match(
        self,
        context: dict,
        *,
        context_strategy: str,
        prompt_variant: str = "region_v4",
        feedback: str | None = None,
    ) -> LLMInvocation: ...

    def explain(self, summary: dict) -> tuple[str, object]: ...


@dataclass
class LLMMatcherConfig:
    context: ContextConfig = field(default_factory=ContextConfig)
    prompt_variant: str = "region_v4"
    max_retries: int = 1
    allow_structural_unknown: bool = True
    explain: bool = True
    capture_transcripts: bool = False


class LLMHeuristicMatcher:
    def __init__(self, client: MatchClient | LLMClient, config: LLMMatcherConfig | None = None) -> None:
        self.client = client
        self.config = config or LLMMatcherConfig()
        self.context_builder = GraphContextBuilder(self.config.context)
        self.mechanics = FingerprintMatcher(
            MatcherConfig(
                max_hops=self.config.context.max_hops,
                max_window_nodes=self.config.context.max_nodes_per_graph * 2,
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

            context = self.context_builder.build(
                left,
                right,
                ready_left,
                ready_right,
                matched_left,
                matched_right,
                value_mapping,
            )
            feedback: str | None = None
            accepted = False
            rejected_anchor_pairs: set[tuple[str, str]] = set()
            rejected_regions = []
            for attempt in range(self.config.max_retries + 1):
                invocation = self.client.propose_match(
                    context,
                    context_strategy=self.config.context.strategy,
                    prompt_variant=self.config.prompt_variant,
                    feedback=feedback,
                )
                invocation.stats.metadata.update(
                    {
                        "attempt": attempt,
                        "accepted": False,
                        "context_chars": len(json.dumps(context, ensure_ascii=False, default=str)),
                        "left_visible_nodes": len(context["left"]["visible_node_ids"]),
                        "right_visible_nodes": len(context["right"]["visible_node_ids"]),
                    }
                )
                if self.config.capture_transcripts:
                    invocation.stats.metadata["raw_response"] = invocation.raw_content
                llm_calls.append(invocation.stats)
                proposal = invocation.proposal
                if proposal is None:
                    feedback = f"The previous response was not valid JSON: {invocation.parse_error}"
                    continue
                if proposal.left_anchor is None or proposal.right_anchor is None:
                    feedback = "No anchor was selected. Select existing IDs only if a defensible rejoin exists."
                    continue
                visible_left = set(context["left"]["visible_node_ids"])
                visible_right = set(context["right"]["visible_node_ids"])
                coarse_pairs = {
                    (item["left"], item["right"]) for item in context.get("coarse_anchor_pairs", [])
                } - rejected_anchor_pairs
                if (proposal.left_anchor, proposal.right_anchor) not in coarse_pairs and (
                    proposal.left_anchor not in visible_left or proposal.right_anchor not in visible_right
                ):
                    feedback = "One or both anchor IDs are absent from the supplied nodes and coarse anchor pairs."
                    continue
                if coarse_pairs and (proposal.left_anchor, proposal.right_anchor) not in coarse_pairs:
                    feedback = (
                        "The proposed anchors are inside an optimization or are not a compatible rejoin pair. "
                        f"Choose one of these coarse anchor pairs: {sorted(coarse_pairs)}"
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
                    feedback = (
                        "The proposed anchors do not form compatible exact anchors with closed mapped entry cuts."
                    )
                    continue
                proof = self.validator.validate(left, right, candidate, value_mapping)
                invocation.stats.metadata.update(
                    {
                        "claimed_rule": proposal.rule_id,
                        "validated_rule": proof.rule_id,
                        "proof_status": proof.status,
                        "anchor_pair": [proposal.left_anchor, proposal.right_anchor],
                        "proposed_left_region": proposal.left_region_nodes,
                        "proposed_right_region": proposal.right_region_nodes,
                        "proposed_exit_mapping": proposal.exit_mapping,
                    }
                )
                if proof.status == "rejected":
                    rejected_anchor_pairs.add((proposal.left_anchor, proposal.right_anchor))
                    rejected_regions.append(candidate)
                    remaining_pairs = {
                        (item["left"], item["right"]) for item in context.get("coarse_anchor_pairs", [])
                    } - rejected_anchor_pairs
                    feedback = "; ".join(proof.details)
                    if not remaining_pairs:
                        break
                    feedback += f" Choose a different remaining pair: {sorted(remaining_pairs)}"
                    continue
                if proof.status == "structural" and not self.config.allow_structural_unknown:
                    feedback = "The region has no supported static proof certificate."
                    continue
                if proof.status == "structural" and any(
                    set(rejected.left_region.node_ids).issubset(candidate.left_region.node_ids)
                    and set(rejected.right_region.node_ids).issubset(candidate.right_region.node_ids)
                    for rejected in rejected_regions
                ):
                    feedback = (
                        "This structural region contains a smaller subregion already rejected by a known certificate."
                    )
                    continue
                if proof.status == "structural" and "same full downstream fingerprint" not in candidate.evidence:
                    feedback = (
                        "An unproven region may be accepted only when the selected anchors have identical full "
                        "downstream fingerprints. Choose an earlier proven anchor or a later stable rejoin."
                    )
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
                        source="llm",
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
                    source="llm",
                    confidence=0.0,
                    explanation=feedback,
                )
            )
            break

        unmatched_left = [node_id for node_id in left.node_order if node_id not in matched_left]
        unmatched_right = [node_id for node_id in right.node_order if node_id not in matched_right]
        result = ComparisonResult(
            method="llm",
            completed=not unmatched_left and not unmatched_right,
            events=events,
            value_mapping=value_mapping,
            unmatched_left=unmatched_left,
            unmatched_right=unmatched_right,
            llm_calls=llm_calls,
            metadata={
                "left_graph": left.name,
                "right_graph": right.name,
                "context_strategy": self.config.context.strategy,
                "prompt_variant": self.config.prompt_variant,
            },
        )
        if self.config.explain:
            summary = build_explanation_summary(left, right, result)
            explanation, stats = self.client.explain(summary)
            result.explanation = explanation
            result.llm_calls.append(stats)  # type: ignore[arg-type]
        return result
