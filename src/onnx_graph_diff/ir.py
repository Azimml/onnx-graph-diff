"""Normalized, hashable intermediate representation of an ONNX graph.

:class:`GraphIR` loads a ``ModelProto`` into a flat, index-preserving view that
the rest of the package can traverse cheaply: nodes keyed by a stable ``node_id``,
values (edges) carrying their producer/consumers, dtype/shape, and any constant
initializer, plus the declared opset imports. Everything is derived once at load
time so matching, fingerprinting, and equivalence checking never have to reparse
the protobuf.

The IR also provides the deterministic primitives those layers build on:

* :meth:`GraphIR.parents` / :meth:`GraphIR.children` -- topological neighbours,
  always returned in graph (index) order.
* :meth:`GraphIR.local_signature` -- an op/attribute/constant hash used for
  exact node matching independent of value names.
* :meth:`GraphIR.descendants` -- a bounded forward window for LLM context.
* :meth:`GraphIR.validate_opset` -- checks that a standard-domain operator is
  actually defined in the graph's declared opset.

Shape inference is attempted on load and silently skipped if it fails, so the IR
is best-effort about shapes but never raises on a model ONNX itself accepts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, ModelProto, TensorProto, helper, numpy_helper


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tensor_summary(tensor: TensorProto) -> dict[str, Any]:
    array = numpy_helper.to_array(tensor)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    summary: dict[str, Any] = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }
    if array.size == 1:
        scalar = array.reshape(()).item()
        summary["scalar"] = scalar.decode("utf-8", errors="replace") if isinstance(scalar, bytes) else scalar
    return summary


def _normalize_attribute(attribute: AttributeProto) -> Any:
    value = helper.get_attribute_value(attribute)
    if isinstance(value, TensorProto):
        return {"tensor": _tensor_summary(value)}
    if isinstance(value, onnx.GraphProto):
        return {"graph": value.name, "nodes": len(value.node)}
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        return [_normalize_attr_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_attr_value(item) for item in value]
    return _normalize_attr_value(value)


def _normalize_attr_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, TensorProto):
        return {"tensor": _tensor_summary(value)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _value_type_shape(value_info: onnx.ValueInfoProto) -> tuple[str | None, tuple[Any, ...] | None]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("elem_type"):
        return None, None
    dtype = TensorProto.DataType.Name(tensor_type.elem_type)
    if not tensor_type.HasField("shape"):
        return dtype, None
    dims: list[Any] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dtype, tuple(dims)


@dataclass(frozen=True)
class InitializerIR:
    name: str
    dtype: str
    shape: tuple[int, ...]
    digest: str
    scalar: Any | None = None
    array: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class ValueIR:
    name: str
    producer: str | None = None
    consumers: list[str] = field(default_factory=list)
    dtype: str | None = None
    shape: tuple[Any, ...] | None = None
    initializer: InitializerIR | None = None
    graph_input: bool = False
    graph_output: bool = False

    def compact(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name}
        if self.dtype is not None:
            result["dtype"] = self.dtype
        if self.shape is not None:
            result["shape"] = list(self.shape)
        if self.initializer is not None:
            result["const"] = {
                "shape": list(self.initializer.shape),
                "dtype": self.initializer.dtype,
                "hash": self.initializer.digest[:16],
            }
            if self.initializer.scalar is not None:
                result["const"]["scalar"] = self.initializer.scalar
        return result


@dataclass(frozen=True)
class NodeIR:
    node_id: str
    name: str
    index: int
    domain: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: tuple[tuple[str, Any], ...]

    @property
    def attrs(self) -> dict[str, Any]:
        return dict(self.attributes)


class GraphIR:
    def __init__(
        self,
        *,
        name: str,
        nodes: dict[str, NodeIR],
        node_order: list[str],
        values: dict[str, ValueIR],
        graph_inputs: list[str],
        graph_outputs: list[str],
        opsets: dict[str, int],
    ) -> None:
        self.name = name
        self.nodes = nodes
        self.node_order = node_order
        self.values = values
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs
        self.opsets = opsets

    @classmethod
    def from_path(cls, path: str, *, infer_shapes: bool = True) -> GraphIR:
        return cls.from_model(onnx.load(path), infer_shapes=infer_shapes)

    @classmethod
    def from_model(cls, model: ModelProto, *, infer_shapes: bool = True) -> GraphIR:
        working = model
        if infer_shapes:
            try:
                working = onnx.shape_inference.infer_shapes(model)
            except (onnx.shape_inference.InferenceError, ValueError, RuntimeError):
                working = model

        graph = working.graph
        values: dict[str, ValueIR] = {}

        for value_info in [*graph.input, *graph.value_info, *graph.output]:
            dtype, shape = _value_type_shape(value_info)
            value = values.setdefault(value_info.name, ValueIR(value_info.name))
            value.dtype = dtype or value.dtype
            value.shape = shape or value.shape

        initializer_names: set[str] = set()
        for tensor in graph.initializer:
            initializer_names.add(tensor.name)
            summary = _tensor_summary(tensor)
            init = InitializerIR(
                name=tensor.name,
                dtype=summary["dtype"],
                shape=tuple(summary["shape"]),
                digest=summary["sha256"],
                scalar=summary.get("scalar"),
                array=np.asarray(numpy_helper.to_array(tensor)),
            )
            value = values.setdefault(tensor.name, ValueIR(tensor.name))
            value.initializer = init
            value.dtype = value.dtype or str(summary["dtype"]).upper()
            value.shape = value.shape or tuple(summary["shape"])

        graph_inputs: list[str] = []
        for value_info in graph.input:
            value = values.setdefault(value_info.name, ValueIR(value_info.name))
            if value_info.name not in initializer_names:
                value.graph_input = True
                graph_inputs.append(value_info.name)

        graph_outputs = [item.name for item in graph.output]
        for name in graph_outputs:
            values.setdefault(name, ValueIR(name)).graph_output = True

        nodes: dict[str, NodeIR] = {}
        node_order: list[str] = []
        seen_ids: set[str] = set()
        for index, node in enumerate(graph.node):
            base_id = node.name or f"node_{index}_{node.op_type}"
            node_id = base_id
            suffix = 1
            while node_id in seen_ids:
                suffix += 1
                node_id = f"{base_id}_{suffix}"
            seen_ids.add(node_id)
            attrs = tuple(sorted((attr.name, _normalize_attribute(attr)) for attr in node.attribute))
            node_ir = NodeIR(
                node_id=node_id,
                name=node.name,
                index=index,
                domain=node.domain or "",
                op_type=node.op_type,
                inputs=tuple(name for name in node.input if name),
                outputs=tuple(name for name in node.output if name),
                attributes=attrs,
            )
            nodes[node_id] = node_ir
            node_order.append(node_id)
            for input_name in node_ir.inputs:
                values.setdefault(input_name, ValueIR(input_name)).consumers.append(node_id)
            for output_name in node_ir.outputs:
                values.setdefault(output_name, ValueIR(output_name)).producer = node_id

        opsets = {item.domain or "": item.version for item in working.opset_import}
        return cls(
            name=graph.name,
            nodes=nodes,
            node_order=node_order,
            values=values,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            opsets=opsets,
        )

    def parents(self, node_id: str) -> tuple[str, ...]:
        parents = {
            producer
            for input_name in self.nodes[node_id].inputs
            if (producer := self.values[input_name].producer) is not None
        }
        return tuple(sorted(parents, key=self.node_index))

    def children(self, node_id: str) -> tuple[str, ...]:
        children: set[str] = set()
        for output_name in self.nodes[node_id].outputs:
            children.update(self.values[output_name].consumers)
        return tuple(sorted(children, key=self.node_index))

    def node_index(self, node_id: str) -> int:
        return self.nodes[node_id].index

    def initializer_signature(self, value_name: str) -> dict[str, Any] | None:
        value = self.values[value_name]
        if value.initializer is None:
            return None
        return {
            "dtype": value.initializer.dtype,
            "shape": value.initializer.shape,
            "digest": value.initializer.digest,
            "scalar": value.initializer.scalar,
        }

    def local_signature(self, node_id: str, *, include_constants: bool = True) -> str:
        node = self.nodes[node_id]
        constant_inputs: list[Any] = []
        if include_constants:
            for position, input_name in enumerate(node.inputs):
                init = self.initializer_signature(input_name)
                if init is not None:
                    constant_inputs.append((position, init))
        outputs = [
            {
                "dtype": self.values[name].dtype,
                "shape": self.values[name].shape,
            }
            for name in node.outputs
        ]
        return _stable_hash(
            {
                "domain": node.domain,
                "op": node.op_type,
                "attrs": node.attributes,
                "inputs": len(node.inputs),
                "outputs": outputs,
                "constants": constant_inputs,
            }
        )

    def compact_node(self, node_id: str, *, include_edges: bool = True) -> dict[str, Any]:
        node = self.nodes[node_id]
        result: dict[str, Any] = {
            "id": node.node_id,
            "op": f"{node.domain}::{node.op_type}" if node.domain else node.op_type,
        }
        if node.attributes:
            result["attrs"] = dict(node.attributes)
        if include_edges:
            result["inputs"] = [self.values[name].compact() for name in node.inputs]
            result["outputs"] = [self.values[name].compact() for name in node.outputs]
            result["parents"] = list(self.parents(node_id))
            result["children"] = list(self.children(node_id))
        return result

    def descendants(self, starts: Iterable[str], *, max_hops: int, max_nodes: int) -> list[str]:
        queue = [(node_id, 0) for node_id in starts]
        seen: set[str] = set()
        result: list[str] = []
        while queue and len(result) < max_nodes:
            node_id, depth = queue.pop(0)
            if node_id in seen or depth > max_hops:
                continue
            seen.add(node_id)
            result.append(node_id)
            if depth < max_hops:
                queue.extend((child, depth + 1) for child in self.children(node_id))
        return sorted(result, key=self.node_index)

    def validate_opset(self, node_id: str) -> bool:
        node = self.nodes[node_id]
        if node.domain not in ("", "ai.onnx"):
            return node.domain in self.opsets
        version = self.opsets.get("", 1)
        try:
            onnx.defs.get_schema(node.op_type, version, "")
        except onnx.onnx_cpp2py_export.defs.SchemaError:
            return False
        return True
