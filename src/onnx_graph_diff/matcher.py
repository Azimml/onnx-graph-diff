"""Deterministic fingerprint matcher and the mechanics shared by all routes.

:class:`FingerprintMatcher` walks both graphs in topological order and repeatedly:

1. commits every node pair whose forward fingerprints are equal (an exact match),
   extending the value mapping across the cut;
2. when the frontier diverges, enumerates candidate rejoin anchors from the
   structural fingerprints, closes the smallest region behind the earliest
   viable anchor, and hands that region to the
   :class:`~onnx_graph_diff.equivalence.StaticEquivalenceValidator`;
3. records a :class:`~onnx_graph_diff.models.MatchEvent` (exact / region /
   unresolved) and continues until both graphs are fully consumed or no anchor
   closes.

The readiness tracking, exact-match commitment, region closure, and candidate
generation implemented here are reused by the LLM and agentic routes -- those
only change *how* the rejoin anchor is chosen, not the surrounding bookkeeping
or the independent equivalence check. This route needs no API key and is fully
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from .equivalence import StaticEquivalenceValidator
from .fingerprint import COMMUTATIVE_OPS, FingerprintIndex
from .ir import GraphIR
from .models import AnchorCandidate, ComparisonResult, MatchEvent, Region


@dataclass
class MatcherConfig:
    max_hops: int = 8
    max_window_nodes: int = 64
    allow_structural_unknown: bool = True


class FingerprintMatcher:
    def __init__(self, config: MatcherConfig | None = None) -> None:
        self.config = config or MatcherConfig()
        self.validator = StaticEquivalenceValidator()

    def compare(self, left: GraphIR, right: GraphIR) -> ComparisonResult:
        left_fp = FingerprintIndex(left)
        right_fp = FingerprintIndex(right)
        value_mapping = self._initial_input_mapping(left, right)
        matched_left: set[str] = set()
        matched_right: set[str] = set()
        events: list[MatchEvent] = []

        while len(matched_left) < len(left.nodes) or len(matched_right) < len(right.nodes):
            ready_left = self._ready(left, matched_left)
            ready_right = self._ready(right, matched_right)
            exact_pairs = self._exact_pairs(
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
                    self._commit_exact(left, right, left_id, right_id, value_mapping, matched_left, matched_right)
                    events.append(
                        MatchEvent(
                            kind="exact",
                            left_nodes=[left_id],
                            right_nodes=[right_id],
                            value_mapping=list(
                                zip(left.nodes[left_id].outputs, right.nodes[right_id].outputs, strict=False)
                            ),
                            source="fingerprint",
                            explanation=(
                                "Exact match accepted because the local node signatures match, every dynamic input "
                                "is already paired by value_mapping, constant-input roles and hashes match, and "
                                "the output dtype/shape descriptors are compatible."
                            ),
                        )
                    )
                continue

            candidates = self.find_anchor_candidates(
                left,
                right,
                ready_left,
                ready_right,
                matched_left,
                matched_right,
                value_mapping,
                left_fp,
                right_fp,
            )
            accepted = False
            validated_candidates = []
            evaluated_candidates = [
                (candidate, self.validator.validate(left, right, candidate, value_mapping)) for candidate in candidates
            ]
            rejected_candidates = [candidate for candidate, proof in evaluated_candidates if proof.status == "rejected"]
            for candidate, proof in evaluated_candidates:
                if proof.status == "rejected":
                    continue
                if proof.status == "structural" and not self.config.allow_structural_unknown:
                    continue
                if proof.status == "structural" and any(
                    set(rejected.left_region.node_ids).issubset(candidate.left_region.node_ids)
                    and set(rejected.right_region.node_ids).issubset(candidate.right_region.node_ids)
                    for rejected in rejected_candidates
                ):
                    continue
                validated_candidates.append((candidate, proof))
            if validated_candidates:
                candidate, proof = max(
                    validated_candidates,
                    key=lambda item: (
                        2 if item[1].status == "proven" else 1,
                        -max(len(item[0].left_region.node_ids), len(item[0].right_region.node_ids)),
                        -len(item[0].left_region.node_ids) - len(item[0].right_region.node_ids),
                        item[0].score,
                    ),
                )
                self._commit_region(candidate, value_mapping, matched_left, matched_right)
                events.append(
                    MatchEvent(
                        kind="region",
                        left_nodes=list(candidate.left_region.node_ids),
                        right_nodes=list(candidate.right_region.node_ids),
                        left_anchor=candidate.left_anchor,
                        right_anchor=candidate.right_anchor,
                        value_mapping=list(candidate.exit_mapping),
                        proof=proof,
                        source="fingerprint",
                        confidence=min(1.0, candidate.score / 150.0),
                        explanation=(
                            "Fingerprint rematch selected this closed region because the anchor local signatures "
                            "match, the region entry values close against the existing value_mapping, and its "
                            f"ranking evidence was: {', '.join(candidate.evidence) or 'local structural compatibility'}."
                        ),
                    )
                )
                accepted = True
            if accepted:
                continue

            events.append(
                MatchEvent(
                    kind="unresolved",
                    left_nodes=ready_left,
                    right_nodes=ready_right,
                    source="fingerprint",
                    confidence=0.0,
                )
            )
            break

        unmatched_left = [node_id for node_id in left.node_order if node_id not in matched_left]
        unmatched_right = [node_id for node_id in right.node_order if node_id not in matched_right]
        completed = not unmatched_left and not unmatched_right
        return ComparisonResult(
            method="fingerprint",
            completed=completed,
            events=events,
            value_mapping=value_mapping,
            unmatched_left=unmatched_left,
            unmatched_right=unmatched_right,
            metadata={
                "left_graph": left.name,
                "right_graph": right.name,
                "allow_structural_unknown": self.config.allow_structural_unknown,
            },
        )

    def _initial_input_mapping(self, left: GraphIR, right: GraphIR) -> dict[str, str]:
        if len(left.graph_inputs) != len(right.graph_inputs):
            return {}
        mapping: dict[str, str] = {}
        for left_name, right_name in zip(left.graph_inputs, right.graph_inputs, strict=False):
            if not self._compatible_values(left, left_name, right, right_name):
                continue
            mapping[left_name] = right_name
        return mapping

    def _ready(self, graph: GraphIR, matched: set[str]) -> list[str]:
        return [
            node_id
            for node_id in graph.node_order
            if node_id not in matched and all(parent in matched for parent in graph.parents(node_id))
        ]

    def _compatible_values(self, left: GraphIR, left_value: str, right: GraphIR, right_value: str) -> bool:
        left_ir = left.values[left_value]
        right_ir = right.values[right_value]
        if left_ir.dtype is not None and right_ir.dtype is not None and left_ir.dtype != right_ir.dtype:
            return False
        if left_ir.shape is not None and right_ir.shape is not None:
            if len(left_ir.shape) != len(right_ir.shape):
                return False
            for left_dim, right_dim in zip(left_ir.shape, right_ir.shape, strict=False):
                if isinstance(left_dim, int) and isinstance(right_dim, int) and left_dim != right_dim:
                    return False
        return True

    def _input_mapping_consistent(
        self,
        left: GraphIR,
        right: GraphIR,
        left_id: str,
        right_id: str,
        value_mapping: dict[str, str],
    ) -> bool:
        left_node = left.nodes[left_id]
        right_node = right.nodes[right_id]
        if len(left_node.inputs) != len(right_node.inputs):
            return False
        if left_node.op_type in COMMUTATIVE_OPS:
            left_dynamic = [name for name in left_node.inputs if left.values[name].initializer is None]
            right_dynamic = [name for name in right_node.inputs if right.values[name].initializer is None]
            mapped = [value_mapping.get(name) for name in left_dynamic]
            return None not in mapped and sorted(mapped) == sorted(right_dynamic)
        for left_name, right_name in zip(left_node.inputs, right_node.inputs, strict=False):
            left_const = left.values[left_name].initializer
            right_const = right.values[right_name].initializer
            if (left_const is None) != (right_const is None):
                return False
            if left_const is not None and right_const is not None:
                if left_const.digest != right_const.digest:
                    return False
            elif value_mapping.get(left_name) != right_name:
                return False
        return True

    def _exact_pairs(
        self,
        left: GraphIR,
        right: GraphIR,
        ready_left: list[str],
        ready_right: list[str],
        value_mapping: dict[str, str],
        left_fp: FingerprintIndex,
        right_fp: FingerprintIndex,
    ) -> list[tuple[str, str]]:
        scored: list[tuple[float, str, str]] = []
        for left_id in ready_left:
            for right_id in ready_right:
                if left_fp[left_id].local != right_fp[right_id].local:
                    continue
                if not self._input_mapping_consistent(left, right, left_id, right_id, value_mapping):
                    continue
                score = 100.0
                if left_fp[left_id].backward == right_fp[right_id].backward:
                    score += 30.0
                score -= abs(left_fp[left_id].output_distance - right_fp[right_id].output_distance)
                scored.append((score, left_id, right_id))
        result: list[tuple[str, str]] = []
        used_left: set[str] = set()
        used_right: set[str] = set()
        for _, left_id, right_id in sorted(scored, reverse=True):
            if left_id in used_left or right_id in used_right:
                continue
            used_left.add(left_id)
            used_right.add(right_id)
            result.append((left_id, right_id))
        return sorted(result, key=lambda pair: left.node_index(pair[0]))

    def _commit_exact(
        self,
        left: GraphIR,
        right: GraphIR,
        left_id: str,
        right_id: str,
        value_mapping: dict[str, str],
        matched_left: set[str],
        matched_right: set[str],
    ) -> None:
        matched_left.add(left_id)
        matched_right.add(right_id)
        for left_output, right_output in zip(left.nodes[left_id].outputs, right.nodes[right_id].outputs, strict=False):
            value_mapping[left_output] = right_output

    def find_anchor_candidates(
        self,
        left: GraphIR,
        right: GraphIR,
        ready_left: list[str],
        ready_right: list[str],
        matched_left: set[str],
        matched_right: set[str],
        value_mapping: dict[str, str],
        left_fp: FingerprintIndex | None = None,
        right_fp: FingerprintIndex | None = None,
    ) -> list[AnchorCandidate]:
        left_fp = left_fp or FingerprintIndex(left)
        right_fp = right_fp or FingerprintIndex(right)
        left_window = left.descendants(
            ready_left,
            max_hops=self.config.max_hops,
            max_nodes=self.config.max_window_nodes,
        )
        right_window = right.descendants(
            ready_right,
            max_hops=self.config.max_hops,
            max_nodes=self.config.max_window_nodes,
        )
        candidates: list[AnchorCandidate] = []
        for left_anchor in left_window:
            for right_anchor in right_window:
                if left_fp[left_anchor].local != right_fp[right_anchor].local:
                    continue
                candidate = self._build_candidate(
                    left,
                    right,
                    left_anchor,
                    right_anchor,
                    matched_left,
                    matched_right,
                    value_mapping,
                    left_fp,
                    right_fp,
                )
                if candidate is not None:
                    candidates.append(candidate)
        for output_index, (left_output, right_output) in enumerate(
            zip(left.graph_outputs, right.graph_outputs, strict=False)
        ):
            candidate = self._build_output_candidate(
                left,
                right,
                output_index,
                left_output,
                right_output,
                matched_left,
                matched_right,
                value_mapping,
            )
            if candidate is not None:
                candidates.append(candidate)
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def build_candidate_from_anchors(
        self,
        left: GraphIR,
        right: GraphIR,
        left_anchor: str,
        right_anchor: str,
        matched_left: set[str],
        matched_right: set[str],
        value_mapping: dict[str, str],
    ) -> AnchorCandidate | None:
        output_prefix = "__graph_output__:"
        if left_anchor.startswith(output_prefix) and right_anchor.startswith(output_prefix):
            try:
                left_index = int(left_anchor.removeprefix(output_prefix))
                right_index = int(right_anchor.removeprefix(output_prefix))
            except ValueError:
                return None
            if (
                left_index != right_index
                or left_index >= len(left.graph_outputs)
                or right_index >= len(right.graph_outputs)
            ):
                return None
            return self._build_output_candidate(
                left,
                right,
                left_index,
                left.graph_outputs[left_index],
                right.graph_outputs[right_index],
                matched_left,
                matched_right,
                value_mapping,
            )
        if left_anchor not in left.nodes or right_anchor not in right.nodes:
            return None
        left_fp = FingerprintIndex(left)
        right_fp = FingerprintIndex(right)
        if left_fp[left_anchor].local != right_fp[right_anchor].local:
            return None
        return self._build_candidate(
            left,
            right,
            left_anchor,
            right_anchor,
            matched_left,
            matched_right,
            value_mapping,
            left_fp,
            right_fp,
        )

    def _collect_region(self, graph: GraphIR, anchor_id: str, matched: set[str]) -> Region:
        exit_values = tuple(
            input_name for input_name in graph.nodes[anchor_id].inputs if graph.values[input_name].initializer is None
        )
        return self._collect_region_for_values(graph, exit_values, matched)

    def _collect_region_for_values(
        self,
        graph: GraphIR,
        exit_values: tuple[str, ...],
        matched: set[str],
    ) -> Region:
        collected: set[str] = set()

        def visit_value(value_name: str) -> None:
            value = graph.values[value_name]
            producer = value.producer
            if producer is None or producer in matched or producer in collected:
                return
            for input_name in graph.nodes[producer].inputs:
                if graph.values[input_name].initializer is None:
                    visit_value(input_name)
            collected.add(producer)

        for value_name in exit_values:
            if graph.values[value_name].initializer is None:
                visit_value(value_name)
        ordered = tuple(sorted(collected, key=graph.node_index))
        entry_values: set[str] = set()
        for node_id in ordered:
            for input_name in graph.nodes[node_id].inputs:
                value = graph.values[input_name]
                if value.initializer is not None:
                    continue
                if value.producer not in collected:
                    entry_values.add(input_name)
        for value_name in exit_values:
            if graph.values[value_name].producer not in collected:
                entry_values.add(value_name)
        return Region(
            node_ids=ordered,
            entry_values=tuple(sorted(entry_values)),
            exit_values=exit_values,
        )

    def _build_output_candidate(
        self,
        left: GraphIR,
        right: GraphIR,
        output_index: int,
        left_output: str,
        right_output: str,
        matched_left: set[str],
        matched_right: set[str],
        value_mapping: dict[str, str],
    ) -> AnchorCandidate | None:
        if value_mapping.get(left_output) == right_output:
            return None
        if not self._compatible_values(left, left_output, right, right_output):
            return None
        left_region = self._collect_region_for_values(left, (left_output,), matched_left)
        right_region = self._collect_region_for_values(right, (right_output,), matched_right)
        if not left_region.node_ids and not right_region.node_ids:
            return None
        expected_right_entries = {value_mapping.get(name) for name in left_region.entry_values}
        if None in expected_right_entries or expected_right_entries != set(right_region.entry_values):
            return None
        anchor = f"__graph_output__:{output_index}"
        return AnchorCandidate(
            left_anchor=anchor,
            right_anchor=anchor,
            left_region=left_region,
            right_region=right_region,
            exit_mapping=((left_output, right_output),),
            score=45.0 - abs(len(left_region.node_ids) - len(right_region.node_ids)) * 0.5,
            evidence=("compatible graph output cut",),
        )

    def _build_candidate(
        self,
        left: GraphIR,
        right: GraphIR,
        left_anchor: str,
        right_anchor: str,
        matched_left: set[str],
        matched_right: set[str],
        value_mapping: dict[str, str],
        left_fp: FingerprintIndex,
        right_fp: FingerprintIndex,
    ) -> AnchorCandidate | None:
        left_node = left.nodes[left_anchor]
        right_node = right.nodes[right_anchor]
        if len(left_node.inputs) != len(right_node.inputs):
            return None
        left_region = self._collect_region(left, left_anchor, matched_left)
        right_region = self._collect_region(right, right_anchor, matched_right)
        if not left_region.node_ids and not right_region.node_ids:
            return None

        expected_right_entries = {value_mapping.get(name) for name in left_region.entry_values}
        if None in expected_right_entries or expected_right_entries != set(right_region.entry_values):
            return None

        exit_mapping: list[tuple[str, str]] = []
        for left_input, right_input in zip(left_node.inputs, right_node.inputs, strict=False):
            left_const = left.values[left_input].initializer
            right_const = right.values[right_input].initializer
            if (left_const is None) != (right_const is None):
                return None
            if left_const is not None and right_const is not None:
                if left_const.digest != right_const.digest:
                    return None
                continue
            if not self._compatible_values(left, left_input, right, right_input):
                return None
            existing = value_mapping.get(left_input)
            if existing is not None and existing != right_input:
                return None
            exit_mapping.append((left_input, right_input))

        evidence: list[str] = ["same local anchor signature"]
        score = 50.0
        if left_fp[left_anchor].backward == right_fp[right_anchor].backward:
            score += 70.0
            evidence.append("same full downstream fingerprint")
        else:
            for depth, bonus in ((4, 30.0), (2, 15.0)):
                if left_fp.downstream_context_hash(left_anchor, depth=depth) == right_fp.downstream_context_hash(
                    right_anchor, depth=depth
                ):
                    score += bonus
                    evidence.append(f"same {depth}-hop downstream context")
                    break
        score -= abs(len(left_region.node_ids) - len(right_region.node_ids)) * 0.5
        score -= abs(left_fp[left_anchor].output_distance - right_fp[right_anchor].output_distance)
        return AnchorCandidate(
            left_anchor=left_anchor,
            right_anchor=right_anchor,
            left_region=left_region,
            right_region=right_region,
            exit_mapping=tuple(exit_mapping),
            score=score,
            evidence=tuple(evidence),
        )

    def _commit_region(
        self,
        candidate: AnchorCandidate,
        value_mapping: dict[str, str],
        matched_left: set[str],
        matched_right: set[str],
    ) -> None:
        matched_left.update(candidate.left_region.node_ids)
        matched_right.update(candidate.right_region.node_ids)
        for left_value, right_value in candidate.exit_mapping:
            value_mapping[left_value] = right_value
