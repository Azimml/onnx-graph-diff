from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper


def _node_id(node: NodeProto, index: int) -> str:
    return node.name or f"node_{index}_{node.op_type}"


def _attribute(node: NodeProto, name: str, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def _producer_map(model: ModelProto) -> dict[str, NodeProto]:
    return {output: node for node in model.graph.node for output in node.output if output}


def _consumer_map(model: ModelProto) -> dict[str, list[NodeProto]]:
    consumers: dict[str, list[NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)
    return consumers


def _initializer_map(model: ModelProto) -> dict[str, TensorProto]:
    return {tensor.name: tensor for tensor in model.graph.initializer}


def _value_shapes(model: ModelProto) -> dict[str, tuple[int | str | None, ...]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except (onnx.shape_inference.InferenceError, RuntimeError, ValueError):
        inferred = model
    shapes: dict[str, tuple[int | str | None, ...]] = {}
    for value_info in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int | str | None] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        shapes[value_info.name] = tuple(dims)
    for tensor in inferred.graph.initializer:
        shapes[tensor.name] = tuple(tensor.dims)
    return shapes


def _graph_output_names(model: ModelProto) -> set[str]:
    return {value.name for value in model.graph.output}


def _replace_all_inputs(model: ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == old:
                node.input[index] = new


def _replace_nodes(model: ModelProto, remove: set[int], insert_at: int, replacements: list[NodeProto]) -> None:
    original = list(model.graph.node)
    rebuilt: list[NodeProto] = []
    inserted = False
    for index, node in enumerate(original):
        if index == insert_at:
            rebuilt.extend(replacements)
            inserted = True
        if index not in remove:
            rebuilt.append(node)
    if not inserted:
        rebuilt.extend(replacements)
    model.graph.ClearField("node")
    model.graph.node.extend(rebuilt)


def _node_index(model: ModelProto, target: NodeProto) -> int:
    for index, node in enumerate(model.graph.node):
        if node is target:
            return index
    for index, node in enumerate(model.graph.node):
        if node.name and node.name == target.name:
            return index
    raise ValueError("Node is absent from graph")


def _unique_name(model: ModelProto, base: str) -> str:
    names = {
        *(node.name for node in model.graph.node if node.name),
        *(value for node in model.graph.node for value in [*node.input, *node.output] if value),
        *(tensor.name for tensor in model.graph.initializer),
    }
    if base not in names:
        return base
    suffix = 2
    while f"{base}_{suffix}" in names:
        suffix += 1
    return f"{base}_{suffix}"


def _single_consumer(consumers: dict[str, list[NodeProto]], value: str) -> NodeProto | None:
    items = consumers.get(value, [])
    return items[0] if len(items) == 1 else None


def _anchor_after(consumers: dict[str, list[NodeProto]], value: str) -> str | None:
    items = consumers.get(value, [])
    if len(items) != 1:
        return None
    return items[0].name or None


def _anchor_after_or_output(
    model: ModelProto,
    consumers: dict[str, list[NodeProto]],
    value: str,
) -> str | None:
    anchor = _anchor_after(consumers, value)
    if anchor is not None:
        return anchor
    for index, graph_output in enumerate(model.graph.output):
        if graph_output.name == value:
            return f"__graph_output__:{index}"
    return None


def _tensor_scalar(initializers: dict[str, TensorProto], name: str) -> float | None:
    tensor = initializers.get(name)
    if tensor is None:
        return None
    array = numpy_helper.to_array(tensor)
    if array.size != 1:
        return None
    return float(array.reshape(()).item())


def _transpose_perm(node: NodeProto, shapes: dict[str, tuple[int | str | None, ...]]) -> list[int] | None:
    perm = _attribute(node, "perm")
    if perm is not None:
        return [int(value) for value in perm]
    shape = shapes.get(node.input[0])
    if shape is None:
        return None
    return list(reversed(range(len(shape))))


def _compose_perm(first: list[int], second: list[int]) -> list[int]:
    return [first[index] for index in second]


def _cleanup_initializers(model: ModelProto) -> None:
    used = {input_name for node in model.graph.node for input_name in node.input if input_name}
    graph_inputs = {value.name for value in model.graph.input}
    kept = [tensor for tensor in model.graph.initializer if tensor.name in used or tensor.name in graph_inputs]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(kept)


@dataclass
class RewriteRecord:
    rule_id: str
    before_nodes: list[str]
    after_nodes: list[str]
    entry_values: list[str]
    left_exit_values: list[str]
    right_exit_values: list[str]
    left_anchor: str | None
    right_anchor: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RewriteOutcome:
    model: ModelProto
    records: list[RewriteRecord]


class RewriteRule(Protocol):
    rule_id: str

    def apply_once(self, model: ModelProto) -> RewriteRecord | None: ...


class IdentityEliminationRule:
    rule_id = "identity_elimination"

    def apply_once(self, model: ModelProto) -> RewriteRecord | None:
        consumers = _consumer_map(model)
        graph_outputs = _graph_output_names(model)
        for index, node in enumerate(model.graph.node):
            if node.domain or node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
                continue
            old_output = node.output[0]
            if old_output in graph_outputs or not consumers.get(old_output):
                continue
            anchor = _anchor_after(consumers, old_output)
            _replace_all_inputs(model, old_output, node.input[0])
            _replace_nodes(model, {index}, index, [])
            return RewriteRecord(
                rule_id=self.rule_id,
                before_nodes=[node.name],
                after_nodes=[],
                entry_values=[node.input[0]],
                left_exit_values=[old_output],
                right_exit_values=[node.input[0]],
                left_anchor=anchor,
                right_anchor=anchor,
            )
        return None


class DoubleTransposeEliminationRule:
    rule_id = "double_transpose_elimination"

    def apply_once(self, model: ModelProto) -> RewriteRecord | None:
        consumers = _consumer_map(model)
        graph_outputs = _graph_output_names(model)
        shapes = _value_shapes(model)
        for first_index, first in enumerate(model.graph.node):
            if first.domain or first.op_type != "Transpose" or len(first.output) != 1:
                continue
            second = _single_consumer(consumers, first.output[0])
            if second is None or second.domain or second.op_type != "Transpose" or len(second.output) != 1:
                continue
            if second.output[0] in graph_outputs:
                continue
            first_perm = _transpose_perm(first, shapes)
            second_perm = _transpose_perm(second, shapes)
            if first_perm is None or second_perm is None or len(first_perm) != len(second_perm):
                continue
            if _compose_perm(first_perm, second_perm) != list(range(len(first_perm))):
                continue
            second_index = _node_index(model, second)
            final_output = second.output[0]
            anchor = _anchor_after(consumers, final_output)
            _replace_all_inputs(model, final_output, first.input[0])
            _replace_nodes(model, {first_index, second_index}, min(first_index, second_index), [])
            return RewriteRecord(
                rule_id=self.rule_id,
                before_nodes=[first.name, second.name],
                after_nodes=[],
                entry_values=[first.input[0]],
                left_exit_values=[final_output],
                right_exit_values=[first.input[0]],
                left_anchor=anchor,
                right_anchor=anchor,
                metadata={"first_perm": first_perm, "second_perm": second_perm},
            )
        return None


class MatMulAddToGemmRule:
    rule_id = "matmul_add_to_gemm"

    def apply_once(self, model: ModelProto) -> RewriteRecord | None:
        consumers = _consumer_map(model)
        initializers = _initializer_map(model)
        _graph_output_names(model)
        shapes = _value_shapes(model)
        for mm_index, mm in enumerate(model.graph.node):
            if mm.domain or mm.op_type != "MatMul" or len(mm.input) != 2 or len(mm.output) != 1:
                continue
            add = _single_consumer(consumers, mm.output[0])
            if add is None or add.domain or add.op_type != "Add" or len(add.input) != 2 or len(add.output) != 1:
                continue
            if shapes.get(mm.input[0]) is None or len(shapes[mm.input[0]]) != 2:
                continue
            if mm.input[1] not in initializers:
                continue
            bias_inputs = [name for name in add.input if name != mm.output[0] and name in initializers]
            if len(bias_inputs) != 1:
                continue
            bias = bias_inputs[0]
            add_index = _node_index(model, add)
            final_output = add.output[0]
            anchor = _anchor_after_or_output(model, consumers, final_output)
            gemm_name = _unique_name(model, f"fused_gemm__{mm.name or mm_index}")
            gemm = helper.make_node(
                "Gemm",
                [mm.input[0], mm.input[1], bias],
                [final_output],
                name=gemm_name,
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=0,
            )
            _replace_nodes(model, {mm_index, add_index}, min(mm_index, add_index), [gemm])
            return RewriteRecord(
                rule_id=self.rule_id,
                before_nodes=[mm.name, add.name],
                after_nodes=[gemm_name],
                entry_values=[mm.input[0], mm.input[1], bias],
                left_exit_values=[final_output],
                right_exit_values=[final_output],
                left_anchor=anchor,
                right_anchor=anchor,
            )
        return None


class GemmDecompositionRule:
    rule_id = "gemm_decomposition"

    def apply_once(self, model: ModelProto) -> RewriteRecord | None:
        consumers = _consumer_map(model)
        _graph_output_names(model)
        for index, node in enumerate(model.graph.node):
            if node.domain or node.op_type != "Gemm" or len(node.input) != 3 or len(node.output) != 1:
                continue
            if float(_attribute(node, "alpha", 1.0)) != 1.0 or float(_attribute(node, "beta", 1.0)) != 1.0:
                continue
            if int(_attribute(node, "transA", 0)) != 0 or int(_attribute(node, "transB", 0)) != 0:
                continue
            final_output = node.output[0]
            anchor = _anchor_after_or_output(model, consumers, final_output)
            mm_output = _unique_name(model, f"{final_output}__matmul")
            mm_name = _unique_name(model, f"decomp_matmul__{node.name or index}")
            add_name = _unique_name(model, f"decomp_add__{node.name or index}")
            mm = helper.make_node("MatMul", [node.input[0], node.input[1]], [mm_output], name=mm_name)
            add = helper.make_node("Add", [mm_output, node.input[2]], [final_output], name=add_name)
            _replace_nodes(model, {index}, index, [mm, add])
            return RewriteRecord(
                rule_id=self.rule_id,
                before_nodes=[node.name],
                after_nodes=[mm_name, add_name],
                entry_values=list(node.input),
                left_exit_values=[final_output],
                right_exit_values=[final_output],
                left_anchor=anchor,
                right_anchor=anchor,
            )
        return None


class ErfGeluFusionRule:
    rule_id = "erf_gelu_fusion"

    def apply_once(self, model: ModelProto) -> RewriteRecord | None:
        if next((item.version for item in model.opset_import if not item.domain), 1) < 20:
            return None
        producers = _producer_map(model)
        consumers = _consumer_map(model)
        initializers = _initializer_map(model)
        _graph_output_names(model)
        for final_index, final_mul in enumerate(model.graph.node):
            if (
                final_mul.domain
                or final_mul.op_type != "Mul"
                or len(final_mul.input) != 2
                or len(final_mul.output) != 1
            ):
                continue
            final_output = final_mul.output[0]
            half_inputs = [name for name in final_mul.input if _tensor_scalar(initializers, name) is not None]
            if len(half_inputs) != 1 or not math.isclose(
                _tensor_scalar(initializers, half_inputs[0]) or 0.0, 0.5, abs_tol=1e-6
            ):
                continue
            mul_x_name = next(name for name in final_mul.input if name != half_inputs[0])
            mul_x = producers.get(mul_x_name)
            if mul_x is None or mul_x.op_type != "Mul" or len(mul_x.input) != 2:
                continue
            add_input = next(
                (name for name in mul_x.input if producers.get(name) is not None and producers[name].op_type == "Add"),
                None,
            )
            if add_input is None:
                continue
            x_name = next(name for name in mul_x.input if name != add_input)
            add = producers[add_input]
            one_inputs = [name for name in add.input if _tensor_scalar(initializers, name) is not None]
            if len(one_inputs) != 1 or not math.isclose(
                _tensor_scalar(initializers, one_inputs[0]) or 0.0, 1.0, abs_tol=1e-6
            ):
                continue
            erf_input = next(name for name in add.input if name != one_inputs[0])
            erf = producers.get(erf_input)
            if erf is None or erf.op_type != "Erf" or len(erf.input) != 1:
                continue
            div = producers.get(erf.input[0])
            if div is None or div.op_type != "Div" or len(div.input) != 2:
                continue
            sqrt_inputs = [name for name in div.input if _tensor_scalar(initializers, name) is not None]
            if len(sqrt_inputs) != 1 or not math.isclose(
                _tensor_scalar(initializers, sqrt_inputs[0]) or 0.0,
                math.sqrt(2.0),
                abs_tol=1e-5,
            ):
                continue
            div_x = next(name for name in div.input if name != sqrt_inputs[0])
            if div_x != x_name:
                continue
            nodes = [div, erf, add, mul_x, final_mul]
            if any(_single_consumer(consumers, node.output[0]) is None for node in nodes[:-1]):
                continue
            indexes = {_node_index(model, node) for node in nodes}
            gelu_name = _unique_name(model, f"fused_gelu__{final_mul.name or final_index}")
            gelu = helper.make_node("Gelu", [x_name], [final_output], name=gelu_name, approximate="none")
            anchor = _anchor_after_or_output(model, consumers, final_output)
            _replace_nodes(model, indexes, min(indexes), [gelu])
            return RewriteRecord(
                rule_id=self.rule_id,
                before_nodes=[node.name for node in nodes],
                after_nodes=[gelu_name],
                entry_values=[x_name],
                left_exit_values=[final_output],
                right_exit_values=[final_output],
                left_anchor=anchor,
                right_anchor=anchor,
            )
        return None


class GeluDecompositionRule:
    rule_id = "gelu_decomposition"

    def apply_once(self, model: ModelProto) -> RewriteRecord | None:
        consumers = _consumer_map(model)
        _graph_output_names(model)
        for index, node in enumerate(model.graph.node):
            if node.domain or node.op_type != "Gelu" or len(node.input) != 1 or len(node.output) != 1:
                continue
            if _attribute(node, "approximate", b"none") not in (b"none", "none"):
                continue
            x_name = node.input[0]
            final_output = node.output[0]
            sqrt_name = _unique_name(model, f"{node.name or index}__sqrt_two")
            one_name = _unique_name(model, f"{node.name or index}__one")
            half_name = _unique_name(model, f"{node.name or index}__half")
            model.graph.initializer.extend(
                [
                    numpy_helper.from_array(np.asarray(math.sqrt(2.0), dtype=np.float32), name=sqrt_name),
                    numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name=one_name),
                    numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), name=half_name),
                ]
            )
            div_out = _unique_name(model, f"{final_output}__div")
            erf_out = _unique_name(model, f"{final_output}__erf")
            add_out = _unique_name(model, f"{final_output}__plus_one")
            mul_out = _unique_name(model, f"{final_output}__mul_x")
            names = [
                _unique_name(model, f"decomp_div__{node.name or index}"),
                _unique_name(model, f"decomp_erf__{node.name or index}"),
                _unique_name(model, f"decomp_add__{node.name or index}"),
                _unique_name(model, f"decomp_mul_x__{node.name or index}"),
                _unique_name(model, f"decomp_mul_half__{node.name or index}"),
            ]
            replacements = [
                helper.make_node("Div", [x_name, sqrt_name], [div_out], name=names[0]),
                helper.make_node("Erf", [div_out], [erf_out], name=names[1]),
                helper.make_node("Add", [erf_out, one_name], [add_out], name=names[2]),
                helper.make_node("Mul", [x_name, add_out], [mul_out], name=names[3]),
                helper.make_node("Mul", [mul_out, half_name], [final_output], name=names[4]),
            ]
            anchor = _anchor_after_or_output(model, consumers, final_output)
            _replace_nodes(model, {index}, index, replacements)
            return RewriteRecord(
                rule_id=self.rule_id,
                before_nodes=[node.name],
                after_nodes=names,
                entry_values=[x_name],
                left_exit_values=[final_output],
                right_exit_values=[final_output],
                left_anchor=anchor,
                right_anchor=anchor,
            )
        return None


class TransposeElementwiseEliminationRule:
    rule_id = "transpose_elementwise_elimination"

    def apply_once(self, model: ModelProto) -> RewriteRecord | None:
        consumers = _consumer_map(model)
        _graph_output_names(model)
        shapes = _value_shapes(model)
        for first_index, first in enumerate(model.graph.node):
            if first.domain or first.op_type != "Transpose" or len(first.output) != 1:
                continue
            relu = _single_consumer(consumers, first.output[0])
            if relu is None or relu.domain or relu.op_type != "Relu" or len(relu.output) != 1:
                continue
            second = _single_consumer(consumers, relu.output[0])
            if second is None or second.domain or second.op_type != "Transpose" or len(second.output) != 1:
                continue
            first_perm = _transpose_perm(first, shapes)
            second_perm = _transpose_perm(second, shapes)
            if first_perm is None or second_perm is None:
                continue
            if _compose_perm(first_perm, second_perm) != list(range(len(first_perm))):
                continue
            indexes = {_node_index(model, item) for item in (first, relu, second)}
            final_output = second.output[0]
            replacement_name = _unique_name(model, f"layout_relu__{relu.name or first_index}")
            replacement = helper.make_node("Relu", [first.input[0]], [final_output], name=replacement_name)
            anchor = _anchor_after_or_output(model, consumers, final_output)
            _replace_nodes(model, indexes, min(indexes), [replacement])
            return RewriteRecord(
                rule_id=self.rule_id,
                before_nodes=[first.name, relu.name, second.name],
                after_nodes=[replacement_name],
                entry_values=[first.input[0]],
                left_exit_values=[final_output],
                right_exit_values=[final_output],
                left_anchor=anchor,
                right_anchor=anchor,
                metadata={"first_perm": first_perm, "second_perm": second_perm},
            )
        return None


class ConvBatchNormFusionRule:
    rule_id = "conv_batchnorm_fusion"

    def apply_once(self, model: ModelProto) -> RewriteRecord | None:
        consumers = _consumer_map(model)
        initializers = _initializer_map(model)
        _graph_output_names(model)
        for conv_index, conv in enumerate(model.graph.node):
            if conv.domain or conv.op_type != "Conv" or len(conv.input) not in (2, 3) or len(conv.output) != 1:
                continue
            bn = _single_consumer(consumers, conv.output[0])
            if (
                bn is None
                or bn.domain
                or bn.op_type != "BatchNormalization"
                or len(bn.input) < 5
                or len(bn.output) != 1
            ):
                continue
            if int(_attribute(bn, "training_mode", 0)) != 0:
                continue
            if any(name not in initializers for name in [conv.input[1], *bn.input[1:5]]):
                continue
            weight = numpy_helper.to_array(initializers[conv.input[1]]).astype(np.float32)
            scale = numpy_helper.to_array(initializers[bn.input[1]]).astype(np.float32)
            bn_bias = numpy_helper.to_array(initializers[bn.input[2]]).astype(np.float32)
            mean = numpy_helper.to_array(initializers[bn.input[3]]).astype(np.float32)
            variance = numpy_helper.to_array(initializers[bn.input[4]]).astype(np.float32)
            if weight.shape[0] != scale.size:
                continue
            if len(conv.input) == 3:
                if conv.input[2] not in initializers:
                    continue
                conv_bias = numpy_helper.to_array(initializers[conv.input[2]]).astype(np.float32)
            else:
                conv_bias = np.zeros_like(scale, dtype=np.float32)
            epsilon = float(_attribute(bn, "epsilon", 1e-5))
            factor = scale / np.sqrt(variance + epsilon)
            fused_weight = weight * factor.reshape((factor.size,) + (1,) * (weight.ndim - 1))
            fused_bias = (conv_bias - mean) * factor + bn_bias
            weight_name = _unique_name(model, f"{conv.name or conv_index}__bn_weight")
            bias_name = _unique_name(model, f"{conv.name or conv_index}__bn_bias")
            model.graph.initializer.extend(
                [
                    numpy_helper.from_array(fused_weight.astype(np.float32), name=weight_name),
                    numpy_helper.from_array(fused_bias.astype(np.float32), name=bias_name),
                ]
            )
            final_output = bn.output[0]
            fused_name = _unique_name(model, f"fused_conv_bn__{conv.name or conv_index}")
            fused = helper.make_node("Conv", [conv.input[0], weight_name, bias_name], [final_output], name=fused_name)
            fused.attribute.extend(copy.deepcopy(conv.attribute))
            bn_index = _node_index(model, bn)
            anchor = _anchor_after_or_output(model, consumers, final_output)
            _replace_nodes(model, {conv_index, bn_index}, min(conv_index, bn_index), [fused])
            return RewriteRecord(
                rule_id=self.rule_id,
                before_nodes=[conv.name, bn.name],
                after_nodes=[fused_name],
                entry_values=list(conv.input) + list(bn.input[1:5]),
                left_exit_values=[final_output],
                right_exit_values=[final_output],
                left_anchor=anchor,
                right_anchor=anchor,
                metadata={"epsilon": epsilon, "fused_weight": weight_name, "fused_bias": bias_name},
            )
        return None


RULES: dict[str, RewriteRule] = {
    rule.rule_id: rule
    for rule in [
        IdentityEliminationRule(),
        DoubleTransposeEliminationRule(),
        MatMulAddToGemmRule(),
        GemmDecompositionRule(),
        ErfGeluFusionRule(),
        GeluDecompositionRule(),
        TransposeElementwiseEliminationRule(),
        ConvBatchNormFusionRule(),
    ]
}


class RuleEngine:
    def __init__(self, rules: dict[str, RewriteRule] | None = None) -> None:
        self.rules = rules or RULES

    def apply(
        self,
        model: ModelProto,
        rule_ids: list[str] | tuple[str, ...],
        *,
        repeat_each_rule: bool = True,
    ) -> RewriteOutcome:
        working = copy.deepcopy(model)
        records: list[RewriteRecord] = []
        for rule_id in rule_ids:
            if rule_id not in self.rules:
                raise KeyError(f"Unknown rewrite rule: {rule_id}")
            while True:
                record = self.rules[rule_id].apply_once(working)
                if record is None:
                    break
                records.append(record)
                _cleanup_initializers(working)
                onnx.checker.check_model(working)
                if not repeat_each_rule:
                    break
        return RewriteOutcome(model=working, records=records)
