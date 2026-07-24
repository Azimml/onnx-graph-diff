"""Merkle-style structural fingerprints over an ONNX DAG.

Each node gets a :class:`NodeFingerprint` with three hashes plus a distance:

* ``local`` -- the node in isolation (op, attributes, constant inputs, output
  types), independent of the names of its edges.
* ``forward`` -- the local hash folded with the fingerprints of everything that
  feeds the node, so two nodes share a forward hash only if their entire input
  cones are structurally identical.
* ``backward`` -- the mirror image: the local hash folded with the fingerprints
  of every consumer, capturing the downstream role of the node.
* ``output_distance`` -- shortest hop count to a graph output, used to bias
  anchor selection toward the earliest clean rejoin.

Inputs to commutative operators (see :data:`COMMUTATIVE_OPS`) are sorted before
hashing so that ``a + b`` and ``b + a`` fingerprint identically. These hashes
drive deterministic exact-matching and rejoin-anchor enumeration in
:mod:`onnx_graph_diff.matcher`; nothing here executes the model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .ir import GraphIR

COMMUTATIVE_OPS = {"Add", "And", "Max", "Min", "Mul", "Or", "Sum", "Xor"}


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class NodeFingerprint:
    local: str
    forward: str
    backward: str
    output_distance: int


class FingerprintIndex:
    """Merkle-style static fingerprints over an ONNX DAG."""

    def __init__(self, graph: GraphIR) -> None:
        self.graph = graph
        self._fingerprints = self._build()

    def __getitem__(self, node_id: str) -> NodeFingerprint:
        return self._fingerprints[node_id]

    def _input_tokens(self, node_id: str, forward: dict[str, str]) -> list[Any]:
        node = self.graph.nodes[node_id]
        tokens: list[Any] = []
        for position, input_name in enumerate(node.inputs):
            value = self.graph.values[input_name]
            if value.initializer is not None:
                token: Any = ("const", value.initializer.digest, value.initializer.shape, value.initializer.dtype)
            elif value.producer is not None:
                token = ("producer", forward[value.producer])
            else:
                token = (
                    "input",
                    value.dtype,
                    value.shape,
                    self.graph.graph_inputs.index(input_name) if input_name in self.graph.graph_inputs else None,
                )
            tokens.append((position, token))
        if node.op_type in COMMUTATIVE_OPS:
            return sorted((token for _, token in tokens), key=str)
        return tokens

    def _output_tokens(self, node_id: str, backward: dict[str, str]) -> list[Any]:
        tokens: list[Any] = []
        node = self.graph.nodes[node_id]
        for output_position, output_name in enumerate(node.outputs):
            value = self.graph.values[output_name]
            if value.graph_output:
                tokens.append((output_position, "graph_output", value.dtype, value.shape))
            for consumer_id in value.consumers:
                consumer = self.graph.nodes[consumer_id]
                input_positions = tuple(index for index, name in enumerate(consumer.inputs) if name == output_name)
                tokens.append((output_position, input_positions, backward[consumer_id]))
        return sorted(tokens, key=str)

    def _build(self) -> dict[str, NodeFingerprint]:
        local = {node_id: self.graph.local_signature(node_id) for node_id in self.graph.node_order}
        forward: dict[str, str] = {}
        for node_id in self.graph.node_order:
            forward[node_id] = _hash((local[node_id], self._input_tokens(node_id, forward)))

        backward: dict[str, str] = {}
        output_distance: dict[str, int] = {}
        for node_id in reversed(self.graph.node_order):
            children = self.graph.children(node_id)
            is_output = any(self.graph.values[name].graph_output for name in self.graph.nodes[node_id].outputs)
            if is_output:
                output_distance[node_id] = 0
            elif children:
                output_distance[node_id] = 1 + min(output_distance[child] for child in children)
            else:
                output_distance[node_id] = 1_000_000
            backward[node_id] = _hash((local[node_id], self._output_tokens(node_id, backward)))

        return {
            node_id: NodeFingerprint(
                local=local[node_id],
                forward=forward[node_id],
                backward=backward[node_id],
                output_distance=output_distance[node_id],
            )
            for node_id in self.graph.node_order
        }

    def downstream_context_hash(self, node_id: str, *, depth: int = 3) -> str:
        memo: dict[tuple[str, int], str] = {}

        def visit(current: str, remaining: int) -> str:
            key = (current, remaining)
            if key in memo:
                return memo[key]
            if remaining == 0:
                result = self[current].local
            else:
                children = [visit(child, remaining - 1) for child in self.graph.children(current)]
                result = _hash((self[current].local, sorted(children)))
            memo[key] = result
            return result

        return visit(node_id, depth)
