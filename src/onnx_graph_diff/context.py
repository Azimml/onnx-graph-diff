"""Serialize bounded windows of the two graphs into the JSON an LLM sees.

:class:`GraphContextBuilder` turns the current mismatch frontier into a compact,
parent-preserving description of both graphs: the frontier nodes, their required
(already-matched) parents, a forward window of successors bounded by
``max_hops``/``max_nodes_per_graph``, the declared opsets, the known value
mapping, and the supported rule catalog. Four :data:`ContextStrategy` values
trade off detail for token budget, from ``parents_only`` up to ``full_window``.

For the assisted ``llm`` route it can also attach ``coarse_anchor_pairs`` --
deterministically computed same-operation candidate anchors (including virtual
``__graph_output__:i`` anchors for terminal rewrites), ranked by distance from
the frontier -- so the model chooses among vetted options rather than inventing
node IDs. A final size guard collapses each node to a minimal descriptor if the
serialized payload would exceed ``max_chars``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .ir import GraphIR

ContextStrategy = Literal["parents_only", "bounded", "path_summary", "full_window"]


@dataclass
class ContextConfig:
    strategy: ContextStrategy = "path_summary"
    max_hops: int = 6
    max_nodes_per_graph: int = 20
    max_chars: int = 24_000
    include_coarse_anchor_pairs: bool = True


class GraphContextBuilder:
    """Build one-shot, parent-preserving mismatch contexts for an LLM."""

    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config = config or ContextConfig()

    def build(
        self,
        left: GraphIR,
        right: GraphIR,
        ready_left: list[str],
        ready_right: list[str],
        matched_left: set[str],
        matched_right: set[str],
        value_mapping: dict[str, str],
    ) -> dict[str, Any]:
        left_side = self._build_side(left, ready_left, matched_left)
        right_side = self._build_side(right, ready_right, matched_right)
        context: dict[str, Any] = {
            "task": "Find the earliest static structural rejoin point after the current mismatch.",
            "constraints": {
                "no_model_execution": True,
                "anchor_nodes_must_exist": True,
                "anchor_nodes_should_match_exactly": True,
                "parents_are_required_context": True,
                "required_parent_nodes_are_already_matched": True,
                "unknown_rules_are_allowed": True,
                "anchor_selection": "When coarse_anchor_pairs is non-empty, choose one of those pairs and prefer the smallest max_hops value.",
            },
            "supported_rule_ids": [
                "identity_elimination",
                "double_transpose_elimination",
                "transpose_elementwise_elimination",
                "matmul_add_to_gemm",
                "erf_gelu_fusion",
                "conv_batchnorm_fusion",
                "unknown_fused_linear",
            ],
            "rule_catalog": [
                {"rule_id": "identity_elimination", "patterns": [["Identity"], []]},
                {"rule_id": "double_transpose_elimination", "patterns": [["Transpose", "Transpose"], []]},
                {
                    "rule_id": "transpose_elementwise_elimination",
                    "patterns": [["Transpose", "Relu", "Transpose"], ["Relu"]],
                },
                {"rule_id": "matmul_add_to_gemm", "patterns": [["MatMul", "Add"], ["Gemm"]]},
                {"rule_id": "erf_gelu_fusion", "patterns": [["Div", "Erf", "Add", "Mul", "Mul"], ["Gelu"]]},
                {"rule_id": "conv_batchnorm_fusion", "patterns": [["Conv", "BatchNormalization"], ["Conv"]]},
            ],
            "opsets": {"left": left.opsets, "right": right.opsets},
            "known_value_mapping": sorted(value_mapping.items()),
            "left": left_side,
            "right": right_side,
        }
        if self.config.include_coarse_anchor_pairs:
            context["coarse_anchor_pairs"] = self._coarse_anchor_pairs(
                left,
                right,
                left_side["visible_node_ids"],
                right_side["visible_node_ids"],
                set(ready_left),
                set(ready_right),
                matched_left,
                matched_right,
            )
        context["context_policy"] = {
            "strategy": self.config.strategy,
            "max_hops": self.config.max_hops,
            "max_nodes_per_graph": self.config.max_nodes_per_graph,
            "required_parent_nodes_are_never_dropped": True,
        }
        return context

    def _build_side(self, graph: GraphIR, frontier: list[str], matched: set[str]) -> dict[str, Any]:
        parent_nodes = sorted(
            {parent for node_id in frontier for parent in graph.parents(node_id)},
            key=graph.node_index,
        )
        parent_values = sorted(
            {
                input_name
                for node_id in frontier
                for input_name in graph.nodes[node_id].inputs
                if graph.values[input_name].initializer is None and graph.values[input_name].producer not in frontier
            }
        )
        required = list(dict.fromkeys([*parent_nodes, *frontier]))
        window = graph.descendants(
            frontier,
            max_hops=self.config.max_hops,
            max_nodes=max(self.config.max_nodes_per_graph * 2, len(required)),
        )
        selected = list(required)
        if self.config.strategy != "parents_only":
            for node_id in window:
                if node_id not in selected and len(selected) < self.config.max_nodes_per_graph:
                    selected.append(node_id)
        else:
            for node_id in window:
                if node_id not in selected and any(parent in frontier for parent in graph.parents(node_id)):
                    selected.append(node_id)
        selected = sorted(set(selected), key=graph.node_index)

        full_ids = set(required)
        if self.config.strategy in ("bounded", "full_window"):
            full_ids = set(selected)
        nodes: list[dict[str, Any]] = []
        for node_id in selected:
            if node_id in full_ids:
                descriptor = self._ssa_node(graph, node_id, matched)
            else:
                node = graph.nodes[node_id]
                descriptor = {
                    "id": node_id,
                    "op": f"{node.domain}::{node.op_type}" if node.domain else node.op_type,
                    "parents": list(graph.parents(node_id)),
                    "output_types": [
                        {"dtype": graph.values[name].dtype, "shape": graph.values[name].shape} for name in node.outputs
                    ],
                }
            nodes.append(descriptor)

        payload = {
            "graph": graph.name,
            "mismatch_frontier": frontier,
            "required_parent_nodes": parent_nodes,
            "required_parent_values": [graph.values[name].compact() for name in parent_values],
            "nodes": nodes,
            "visible_node_ids": selected,
            "omitted_node_count": max(0, len(window) - len(selected)),
        }
        if len(json.dumps(payload, ensure_ascii=True, default=str)) > self.config.max_chars:
            payload["nodes"] = [self._minimal_node(graph, node_id) for node_id in selected]
            payload["compacted_for_char_budget"] = True
        return payload

    def _ssa_node(self, graph: GraphIR, node_id: str, matched: set[str]) -> dict[str, Any]:
        node = graph.nodes[node_id]
        inputs: list[dict[str, Any]] = []
        for name in node.inputs:
            value = graph.values[name]
            item = value.compact()
            if value.initializer is not None:
                item["from"] = "initializer"
            elif value.producer is not None:
                item["from"] = value.producer
            elif value.graph_input:
                item["from"] = "graph_input"
            else:
                item["from"] = "external"
            inputs.append(item)
        return {
            "id": node_id,
            "op": f"{node.domain}::{node.op_type}" if node.domain else node.op_type,
            "attrs": dict(node.attributes),
            "inputs": inputs,
            "outputs": [graph.values[name].compact() for name in node.outputs],
            "parents": list(graph.parents(node_id)),
            "children": list(graph.children(node_id)),
            "matched_parent_flags": {parent: parent in matched for parent in graph.parents(node_id)},
        }

    def _minimal_node(self, graph: GraphIR, node_id: str) -> dict[str, Any]:
        node = graph.nodes[node_id]
        return {
            "id": node_id,
            "op": f"{node.domain}::{node.op_type}" if node.domain else node.op_type,
            "parents": list(graph.parents(node_id)),
            "inputs": [
                {
                    "name": name,
                    "from": graph.values[name].producer
                    or ("initializer" if graph.values[name].initializer else "graph_input"),
                }
                for name in node.inputs
            ],
            "outputs": list(node.outputs),
        }

    def _coarse_anchor_pairs(
        self,
        left: GraphIR,
        right: GraphIR,
        left_ids: list[str],
        right_ids: list[str],
        left_frontier: set[str],
        right_frontier: set[str],
        matched_left: set[str],
        matched_right: set[str],
    ) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for left_id in left_ids:
            for right_id in right_ids:
                if left_id in left_frontier and right_id in right_frontier:
                    continue
                left_node = left.nodes[left_id]
                right_node = right.nodes[right_id]
                if (left_node.domain, left_node.op_type, len(left_node.inputs), len(left_node.outputs)) != (
                    right_node.domain,
                    right_node.op_type,
                    len(right_node.inputs),
                    len(right_node.outputs),
                ):
                    continue
                if left.local_signature(left_id) != right.local_signature(right_id):
                    continue
                shape_compatible = all(
                    self._shape_compatible(left.values[lv].shape, right.values[rv].shape)
                    for lv, rv in zip(left_node.outputs, right_node.outputs, strict=False)
                )
                if shape_compatible:
                    left_hops = self._distance_from_frontier(left, left_frontier, left_id)
                    right_hops = self._distance_from_frontier(right, right_frontier, right_id)
                    pairs.append(
                        {
                            "left": left_id,
                            "right": right_id,
                            "left_hops": left_hops,
                            "right_hops": right_hops,
                            "max_hops": max(left_hops, right_hops),
                            "left_region_nodes": self._candidate_region(left, left_id, matched_left),
                            "right_region_nodes": self._candidate_region(right, right_id, matched_right),
                        }
                    )
        for output_index, (left_output, right_output) in enumerate(
            zip(left.graph_outputs, right.graph_outputs, strict=False)
        ):
            if not self._shape_compatible(left.values[left_output].shape, right.values[right_output].shape):
                continue
            left_region = self._candidate_region_values(left, [left_output], matched_left)
            right_region = self._candidate_region_values(right, [right_output], matched_right)
            if not left_region and not right_region:
                continue
            left_producer = left.values[left_output].producer
            right_producer = right.values[right_output].producer
            left_hops = self._distance_from_frontier(left, left_frontier, left_producer) + 1 if left_producer else 0
            right_hops = (
                self._distance_from_frontier(right, right_frontier, right_producer) + 1 if right_producer else 0
            )
            anchor = f"__graph_output__:{output_index}"
            pairs.append(
                {
                    "left": anchor,
                    "right": anchor,
                    "left_hops": left_hops,
                    "right_hops": right_hops,
                    "max_hops": max(left_hops, right_hops),
                    "left_region_nodes": left_region,
                    "right_region_nodes": right_region,
                    "virtual_output": True,
                }
            )
        return sorted(pairs, key=lambda item: (item["max_hops"], item["left_hops"] + item["right_hops"]))[:64]

    @staticmethod
    def _distance_from_frontier(graph: GraphIR, frontier: set[str], target: str | None) -> int:
        if target is None:
            return 1_000_000
        queue = [(node_id, 0) for node_id in frontier]
        seen: set[str] = set()
        while queue:
            node_id, distance = queue.pop(0)
            if node_id in seen:
                continue
            seen.add(node_id)
            if node_id == target:
                return distance
            queue.extend((child, distance + 1) for child in graph.children(node_id))
        return 1_000_000

    @staticmethod
    def _candidate_region(graph: GraphIR, anchor_id: str, matched: set[str]) -> list[str]:
        exit_values = [
            input_name for input_name in graph.nodes[anchor_id].inputs if graph.values[input_name].initializer is None
        ]
        return GraphContextBuilder._candidate_region_values(graph, exit_values, matched)

    @staticmethod
    def _candidate_region_values(graph: GraphIR, exit_values: list[str], matched: set[str]) -> list[str]:
        collected: set[str] = set()

        def visit_value(value_name: str) -> None:
            producer = graph.values[value_name].producer
            if producer is None or producer in matched or producer in collected:
                return
            for input_name in graph.nodes[producer].inputs:
                if graph.values[input_name].initializer is None:
                    visit_value(input_name)
            collected.add(producer)

        for value_name in exit_values:
            if graph.values[value_name].initializer is None:
                visit_value(value_name)
        return sorted(collected, key=graph.node_index)

    @staticmethod
    def _shape_compatible(left_shape: tuple[Any, ...] | None, right_shape: tuple[Any, ...] | None) -> bool:
        if left_shape is None or right_shape is None:
            return True
        if len(left_shape) != len(right_shape):
            return False
        return all(
            not (isinstance(left_dim, int) and isinstance(right_dim, int) and left_dim != right_dim)
            for left_dim, right_dim in zip(left_shape, right_shape, strict=False)
        )
