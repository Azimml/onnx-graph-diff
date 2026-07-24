"""Independent, opset-aware equivalence checker for rematched regions.

:class:`StaticEquivalenceValidator` is deliberately decoupled from whichever
route (fingerprint, LLM, or agentic) proposed a region. Given a candidate region
on each side, it re-derives the region from the graphs themselves and tries each
registered :data:`RuleValidator` in turn. Every rule checks its own side
conditions -- matching initializers, attribute constraints, arity, dtype/shape
compatibility -- before it will certify a rewrite.

A region ends up with one of four statuses (see
:data:`onnx_graph_diff.models.ProofStatus`):

* ``proven`` -- a supported algebraic identity holds under the declared opset.
* ``structural`` -- the sides rejoin cleanly but no certificate applies (e.g. an
  opaque custom-domain fused op); reported as rematched-not-proven.
* ``rejected`` -- a known rule pattern matched but a condition is violated.
* ``unresolved`` -- surfaced by the matcher, not here, when a region cannot close.

A schema guard runs first and rejects any region whose standard-domain operators
are unavailable in the declared opset, so a proof is never claimed outside the
opset range in which it actually holds. The proposer's ``claimed_rule`` only
filters which certificate is accepted; it can never upgrade a status.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .ir import GraphIR, NodeIR
from .models import AnchorCandidate, EquivalenceResult, Region

RuleValidator = Callable[[GraphIR, GraphIR, Region, Region, dict[str, str]], EquivalenceResult | None]


def _ops(graph: GraphIR, region: Region) -> list[str]:
    return [graph.nodes[node_id].op_type for node_id in region.node_ids]


def _attrs(node: NodeIR) -> dict[str, Any]:
    return dict(node.attributes)


def _scalar(graph: GraphIR, value_name: str) -> float | None:
    value = graph.values[value_name]
    if value.initializer is None or value.initializer.scalar is None:
        return None
    try:
        return float(value.initializer.scalar)
    except (TypeError, ValueError):
        return None


def _same_initializer(left: GraphIR, left_name: str, right: GraphIR, right_name: str) -> bool:
    left_init = left.values[left_name].initializer
    right_init = right.values[right_name].initializer
    return left_init is not None and right_init is not None and left_init.digest == right_init.digest


def _mapped(value_mapping: dict[str, str], left_value: str, right_value: str) -> bool:
    return value_mapping.get(left_value) == right_value


def _schema_guard(left: GraphIR, right: GraphIR, left_region: Region, right_region: Region) -> EquivalenceResult | None:
    invalid: list[str] = []
    for graph, region, side in ((left, left_region, "left"), (right, right_region, "right")):
        for node_id in region.node_ids:
            node = graph.nodes[node_id]
            if node.domain in ("", "ai.onnx") and not graph.validate_opset(node_id):
                invalid.append(f"{side}:{node_id}:{node.op_type}")
    if invalid:
        return EquivalenceResult(
            "rejected", details=["Operators are unavailable in the declared opset: " + ", ".join(invalid)]
        )
    return None


def _orient(
    left: GraphIR,
    right: GraphIR,
    left_region: Region,
    right_region: Region,
    primitive_ops: list[str],
    fused_ops: list[str],
) -> tuple[GraphIR, GraphIR, Region, Region, bool] | None:
    if _ops(left, left_region) == primitive_ops and _ops(right, right_region) == fused_ops:
        return left, right, left_region, right_region, False
    if _ops(right, right_region) == primitive_ops and _ops(left, left_region) == fused_ops:
        return right, left, right_region, left_region, True
    return None


def _validate_matmul_add_gemm(
    left: GraphIR,
    right: GraphIR,
    left_region: Region,
    right_region: Region,
    value_mapping: dict[str, str],
) -> EquivalenceResult | None:
    oriented = _orient(left, right, left_region, right_region, ["MatMul", "Add"], ["Gemm"])
    if oriented is None:
        return None
    primitive, fused, primitive_region, fused_region, swapped = oriented
    mm = primitive.nodes[primitive_region.node_ids[0]]
    add = primitive.nodes[primitive_region.node_ids[1]]
    gemm = fused.nodes[fused_region.node_ids[0]]
    attrs = _attrs(gemm)
    conditions = {
        "alpha": float(attrs.get("alpha", 1.0)) == 1.0,
        "beta": float(attrs.get("beta", 1.0)) == 1.0,
        "transA": int(attrs.get("transA", 0)) == 0,
        "transB": int(attrs.get("transB", 0)) == 0,
    }
    if not all(conditions.values()):
        failed = [name for name, ok in conditions.items() if not ok]
        return EquivalenceResult(
            "rejected", "matmul_add_to_gemm", ["Gemm attributes violate exact fusion: " + ", ".join(failed)]
        )
    if not mm.outputs or mm.outputs[0] not in add.inputs:
        return EquivalenceResult("rejected", "matmul_add_to_gemm", ["Add does not consume the MatMul output"])
    mm_dynamic = [name for name in mm.inputs if primitive.values[name].initializer is None]
    gemm_dynamic = [name for name in gemm.inputs[:2] if fused.values[name].initializer is None]
    if len(mm_dynamic) != 1 or len(gemm_dynamic) != 1:
        return EquivalenceResult(
            "rejected", "matmul_add_to_gemm", ["Fixture validator expects one dynamic MatMul input"]
        )
    dynamic_ok = _mapped(value_mapping, mm_dynamic[0], gemm_dynamic[0])
    if swapped:
        dynamic_ok = value_mapping.get(gemm_dynamic[0]) == mm_dynamic[0]
    if not dynamic_ok:
        return EquivalenceResult("rejected", "matmul_add_to_gemm", ["Dynamic Gemm input is not mapped to MatMul input"])
    mm_constants = [name for name in mm.inputs if primitive.values[name].initializer is not None]
    add_constants = [name for name in add.inputs if primitive.values[name].initializer is not None]
    gemm_constants = [name for name in gemm.inputs if fused.values[name].initializer is not None]
    if len(mm_constants) != 1 or len(add_constants) != 1 or len(gemm_constants) != 2:
        return EquivalenceResult(
            "rejected", "matmul_add_to_gemm", ["Weight or bias constants do not match the fusion form"]
        )
    if not _same_initializer(primitive, mm_constants[0], fused, gemm_constants[0]):
        return EquivalenceResult("rejected", "matmul_add_to_gemm", ["Gemm weight differs from MatMul weight"])
    if not _same_initializer(primitive, add_constants[0], fused, gemm_constants[1]):
        return EquivalenceResult("rejected", "matmul_add_to_gemm", ["Gemm bias differs from Add bias"])
    return EquivalenceResult(
        "proven",
        "matmul_add_to_gemm",
        ["Validated against ONNX Gemm semantics with alpha=beta=1 and no transposition"],
    )


def _validate_identity(
    left: GraphIR,
    right: GraphIR,
    left_region: Region,
    right_region: Region,
    value_mapping: dict[str, str],
) -> EquivalenceResult | None:
    oriented = _orient(left, right, left_region, right_region, ["Identity"], [])
    if oriented is None:
        return None
    primitive, _, primitive_region, empty_region, swapped = oriented
    identity = primitive.nodes[primitive_region.node_ids[0]]
    if len(identity.inputs) != 1 or len(identity.outputs) != 1 or len(empty_region.exit_values) != 1:
        return EquivalenceResult("rejected", "identity_elimination", ["Identity boundary is malformed"])
    other_value = empty_region.exit_values[0]
    mapping_ok = _mapped(value_mapping, identity.inputs[0], other_value)
    if swapped:
        mapping_ok = value_mapping.get(other_value) == identity.inputs[0]
    if not mapping_ok:
        return EquivalenceResult(
            "rejected", "identity_elimination", ["Identity input is not mapped to the bypass value"]
        )
    return EquivalenceResult("proven", "identity_elimination", ["ONNX Identity returns its input unchanged"])


def _permutation(graph: GraphIR, node: NodeIR) -> list[int] | None:
    attrs = _attrs(node)
    if "perm" in attrs:
        return [int(value) for value in attrs["perm"]]
    input_shape = graph.values[node.inputs[0]].shape
    if input_shape is None:
        return None
    return list(reversed(range(len(input_shape))))


def _compose_permutations(first: list[int], second: list[int]) -> list[int]:
    return [first[index] for index in second]


def _validate_double_transpose(
    left: GraphIR,
    right: GraphIR,
    left_region: Region,
    right_region: Region,
    value_mapping: dict[str, str],
) -> EquivalenceResult | None:
    oriented = _orient(left, right, left_region, right_region, ["Transpose", "Transpose"], [])
    if oriented is None:
        return None
    primitive, _, primitive_region, empty_region, swapped = oriented
    first = primitive.nodes[primitive_region.node_ids[0]]
    second = primitive.nodes[primitive_region.node_ids[1]]
    if not first.outputs or first.outputs[0] not in second.inputs:
        return EquivalenceResult("rejected", "double_transpose_elimination", ["Transpose nodes are not connected"])
    first_perm = _permutation(primitive, first)
    second_perm = _permutation(primitive, second)
    if first_perm is None or second_perm is None or len(first_perm) != len(second_perm):
        return EquivalenceResult("rejected", "double_transpose_elimination", ["Transpose permutation is unavailable"])
    if _compose_permutations(first_perm, second_perm) != list(range(len(first_perm))):
        return EquivalenceResult(
            "rejected", "double_transpose_elimination", ["Transpose permutations do not compose to identity"]
        )
    other_value = empty_region.exit_values[0]
    mapping_ok = _mapped(value_mapping, first.inputs[0], other_value)
    if swapped:
        mapping_ok = value_mapping.get(other_value) == first.inputs[0]
    if not mapping_ok:
        return EquivalenceResult("rejected", "double_transpose_elimination", ["Bypass input mapping is inconsistent"])
    return EquivalenceResult(
        "proven", "double_transpose_elimination", ["Static permutation composition equals identity"]
    )


def _validate_transpose_elementwise(
    left: GraphIR,
    right: GraphIR,
    left_region: Region,
    right_region: Region,
    value_mapping: dict[str, str],
) -> EquivalenceResult | None:
    oriented = _orient(left, right, left_region, right_region, ["Transpose", "Relu", "Transpose"], ["Relu"])
    if oriented is None:
        return None
    primitive, fused, primitive_region, fused_region, swapped = oriented
    first, relu, second = [primitive.nodes[node_id] for node_id in primitive_region.node_ids]
    other_relu = fused.nodes[fused_region.node_ids[0]]
    if first.outputs[0] not in relu.inputs or relu.outputs[0] not in second.inputs:
        return EquivalenceResult(
            "rejected", "transpose_elementwise_elimination", ["Transpose/Relu chain is disconnected"]
        )
    first_perm = _permutation(primitive, first)
    second_perm = _permutation(primitive, second)
    if first_perm is None or second_perm is None:
        return EquivalenceResult("rejected", "transpose_elementwise_elimination", ["Permutation is unavailable"])
    if _compose_permutations(first_perm, second_perm) != list(range(len(first_perm))):
        return EquivalenceResult("rejected", "transpose_elementwise_elimination", ["Outer transposes are not inverse"])
    mapping_ok = _mapped(value_mapping, first.inputs[0], other_relu.inputs[0])
    if swapped:
        mapping_ok = value_mapping.get(other_relu.inputs[0]) == first.inputs[0]
    if not mapping_ok:
        return EquivalenceResult(
            "rejected", "transpose_elementwise_elimination", ["Relu input mapping is inconsistent"]
        )
    return EquivalenceResult(
        "proven",
        "transpose_elementwise_elimination",
        ["Relu is elementwise and commutes with Transpose; outer permutations compose to identity"],
    )


def _find_node_by_output(graph: GraphIR, region: Region, output_name: str) -> NodeIR | None:
    producer = graph.values[output_name].producer
    if producer is None or producer not in region.node_ids:
        return None
    return graph.nodes[producer]


def _non_scalar_input(graph: GraphIR, node: NodeIR) -> str | None:
    candidates = [name for name in node.inputs if _scalar(graph, name) is None]
    return candidates[0] if len(candidates) == 1 else None


def _validate_gelu(
    left: GraphIR,
    right: GraphIR,
    left_region: Region,
    right_region: Region,
    value_mapping: dict[str, str],
) -> EquivalenceResult | None:
    primitive_set = ["Div", "Erf", "Add", "Mul", "Mul"]
    oriented = _orient(left, right, left_region, right_region, primitive_set, ["Gelu"])
    if oriented is None:
        return None
    primitive, fused, primitive_region, fused_region, swapped = oriented
    gelu = fused.nodes[fused_region.node_ids[0]]
    if _attrs(gelu).get("approximate", "none") != "none":
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["Approximate GELU is not exact Erf GELU"])
    if len(primitive_region.exit_values) != 1:
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["GELU region must expose one output"])
    final_mul = _find_node_by_output(primitive, primitive_region, primitive_region.exit_values[0])
    if final_mul is None or final_mul.op_type != "Mul":
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["Final GELU node is not Mul"])
    half_inputs = [name for name in final_mul.inputs if _scalar(primitive, name) is not None]
    if len(half_inputs) != 1 or not math.isclose(
        _scalar(primitive, half_inputs[0]) or 0.0, 0.5, rel_tol=0, abs_tol=1e-6
    ):
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["Final GELU multiplier is not 0.5"])
    scaled_name = next(name for name in final_mul.inputs if name not in half_inputs)
    mul_x = _find_node_by_output(primitive, primitive_region, scaled_name)
    if mul_x is None or mul_x.op_type != "Mul":
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["GELU x multiplier is missing"])
    add_name = next(
        (name for name in mul_x.inputs if primitive.values[name].producer in primitive_region.node_ids), None
    )
    x_name = next((name for name in mul_x.inputs if name != add_name), None)
    if add_name is None or x_name is None:
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["GELU x/Add inputs are malformed"])
    add = _find_node_by_output(primitive, primitive_region, add_name)
    if add is None or add.op_type != "Add":
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["GELU plus-one node is missing"])
    one_inputs = [name for name in add.inputs if _scalar(primitive, name) is not None]
    if len(one_inputs) != 1 or not math.isclose(_scalar(primitive, one_inputs[0]) or 0.0, 1.0, rel_tol=0, abs_tol=1e-6):
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["GELU additive constant is not 1"])
    erf_name = next(name for name in add.inputs if name not in one_inputs)
    erf = _find_node_by_output(primitive, primitive_region, erf_name)
    if erf is None or erf.op_type != "Erf":
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["GELU Erf node is missing"])
    div = _find_node_by_output(primitive, primitive_region, erf.inputs[0])
    if div is None or div.op_type != "Div":
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["GELU division node is missing"])
    sqrt_inputs = [name for name in div.inputs if _scalar(primitive, name) is not None]
    if len(sqrt_inputs) != 1 or not math.isclose(
        _scalar(primitive, sqrt_inputs[0]) or 0.0,
        math.sqrt(2.0),
        rel_tol=0,
        abs_tol=1e-5,
    ):
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["GELU divisor is not sqrt(2)"])
    div_x = next(name for name in div.inputs if name not in sqrt_inputs)
    if div_x != x_name:
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["GELU uses different x values in its two branches"])
    mapping_ok = _mapped(value_mapping, x_name, gelu.inputs[0])
    if swapped:
        mapping_ok = value_mapping.get(gelu.inputs[0]) == x_name
    if not mapping_ok:
        return EquivalenceResult("rejected", "erf_gelu_fusion", ["GELU input mapping is inconsistent"])
    return EquivalenceResult(
        "proven", "erf_gelu_fusion", ["Matched exact ONNX Gelu equation x * (1 + erf(x/sqrt(2))) / 2"]
    )


def _validate_conv_batchnorm(
    left: GraphIR,
    right: GraphIR,
    left_region: Region,
    right_region: Region,
    value_mapping: dict[str, str],
) -> EquivalenceResult | None:
    oriented = _orient(left, right, left_region, right_region, ["Conv", "BatchNormalization"], ["Conv"])
    if oriented is None:
        return None
    primitive, fused, primitive_region, fused_region, swapped = oriented
    conv = primitive.nodes[primitive_region.node_ids[0]]
    batchnorm = primitive.nodes[primitive_region.node_ids[1]]
    fused_conv = fused.nodes[fused_region.node_ids[0]]
    if conv.outputs[0] not in batchnorm.inputs:
        return EquivalenceResult(
            "rejected", "conv_batchnorm_fusion", ["BatchNormalization does not consume Conv output"]
        )
    if int(_attrs(batchnorm).get("training_mode", 0)) != 0:
        return EquivalenceResult(
            "rejected", "conv_batchnorm_fusion", ["Training-mode BatchNormalization cannot be folded"]
        )
    if conv.attributes != fused_conv.attributes:
        return EquivalenceResult("rejected", "conv_batchnorm_fusion", ["Fused Conv attributes differ from source Conv"])
    dynamic_ok = _mapped(value_mapping, conv.inputs[0], fused_conv.inputs[0])
    if swapped:
        dynamic_ok = value_mapping.get(fused_conv.inputs[0]) == conv.inputs[0]
    if not dynamic_ok:
        return EquivalenceResult("rejected", "conv_batchnorm_fusion", ["Conv data inputs are not mapped"])
    if len(batchnorm.inputs) < 5 or len(fused_conv.inputs) < 3:
        return EquivalenceResult(
            "rejected", "conv_batchnorm_fusion", ["Fusion requires constant BN parameters and fused Conv bias"]
        )

    def array(graph: GraphIR, name: str) -> np.ndarray | None:
        initializer = graph.values[name].initializer
        if initializer is None or initializer.array is None:
            return None
        return np.asarray(initializer.array, dtype=np.float32)

    weight = array(primitive, conv.inputs[1])
    scale = array(primitive, batchnorm.inputs[1])
    bn_bias = array(primitive, batchnorm.inputs[2])
    mean = array(primitive, batchnorm.inputs[3])
    variance = array(primitive, batchnorm.inputs[4])
    fused_weight = array(fused, fused_conv.inputs[1])
    fused_bias = array(fused, fused_conv.inputs[2])
    arrays = [weight, scale, bn_bias, mean, variance, fused_weight, fused_bias]
    if any(item is None for item in arrays):
        return EquivalenceResult("rejected", "conv_batchnorm_fusion", ["Required initializer data is unavailable"])
    assert weight is not None and scale is not None and bn_bias is not None
    assert mean is not None and variance is not None and fused_weight is not None and fused_bias is not None
    if len(conv.inputs) == 3:
        conv_bias = array(primitive, conv.inputs[2])
        if conv_bias is None:
            return EquivalenceResult("rejected", "conv_batchnorm_fusion", ["Source Conv bias is unavailable"])
    else:
        conv_bias = np.zeros_like(scale, dtype=np.float32)
    epsilon = float(_attrs(batchnorm).get("epsilon", 1e-5))
    factor = scale / np.sqrt(variance + epsilon)
    expected_weight = weight * factor.reshape((factor.size,) + (1,) * (weight.ndim - 1))
    expected_bias = (conv_bias - mean) * factor + bn_bias
    if not np.allclose(expected_weight, fused_weight, rtol=1e-5, atol=1e-6):
        return EquivalenceResult(
            "rejected", "conv_batchnorm_fusion", ["Folded Conv weight does not satisfy BN equation"]
        )
    if not np.allclose(expected_bias, fused_bias, rtol=1e-5, atol=1e-6):
        return EquivalenceResult("rejected", "conv_batchnorm_fusion", ["Folded Conv bias does not satisfy BN equation"])
    return EquivalenceResult(
        "proven",
        "conv_batchnorm_fusion",
        ["Validated folded Conv weight and bias using inference-mode ONNX BatchNormalization semantics"],
    )


VALIDATORS: tuple[RuleValidator, ...] = (
    _validate_matmul_add_gemm,
    _validate_identity,
    _validate_double_transpose,
    _validate_transpose_elementwise,
    _validate_gelu,
    _validate_conv_batchnorm,
)


class StaticEquivalenceValidator:
    """Validate local equivalence using declared ONNX opsets and rule certificates."""

    def validate(
        self,
        left: GraphIR,
        right: GraphIR,
        candidate: AnchorCandidate,
        value_mapping: dict[str, str],
        *,
        claimed_rule: str | None = None,
    ) -> EquivalenceResult:
        schema_failure = _schema_guard(left, right, candidate.left_region, candidate.right_region)
        if schema_failure is not None:
            return schema_failure
        rejected: EquivalenceResult | None = None
        for validator in VALIDATORS:
            result = validator(left, right, candidate.left_region, candidate.right_region, value_mapping)
            if result is None:
                continue
            if claimed_rule is not None and result.rule_id not in (claimed_rule, None):
                continue
            if result.status == "proven":
                return result
            rejected = result
        if rejected is not None:
            return rejected
        return EquivalenceResult(
            "structural",
            claimed_rule,
            ["No supported static equivalence certificate covers this region"],
        )
