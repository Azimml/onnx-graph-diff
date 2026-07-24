from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from onnx_graph_diff.equivalence import StaticEquivalenceValidator
from onnx_graph_diff.ir import GraphIR
from onnx_graph_diff.models import AnchorCandidate, Region


def _candidate(left_region: Region, right_region: Region) -> AnchorCandidate:
    exit_mapping = tuple(zip(left_region.exit_values, right_region.exit_values, strict=False))
    return AnchorCandidate(
        left_anchor="left_relu",
        right_anchor="right_relu",
        left_region=left_region,
        right_region=right_region,
        exit_mapping=exit_mapping,
        score=1.0,
    )


def test_matmul_add_to_gemm_is_proven(corpus_dir):
    left = GraphIR.from_path(str(corpus_dir / "matmul_add_gemm.left.onnx"))
    right = GraphIR.from_path(str(corpus_dir / "matmul_add_gemm.right.onnx"))
    candidate = _candidate(
        Region(("left_matmul", "left_add"), ("x",), ("pre_relu",)),
        Region(("right_gemm",), ("x",), ("pre_relu",)),
    )
    result = StaticEquivalenceValidator().validate(left, right, candidate, {"x": "x"})
    assert result.status == "proven"
    assert result.rule_id == "matmul_add_to_gemm"


def test_opaque_custom_op_is_structural_not_proven(corpus_dir):
    left = GraphIR.from_path(str(corpus_dir / "unknown_fused_linear.left.onnx"))
    right = GraphIR.from_path(str(corpus_dir / "unknown_fused_linear.right.onnx"))
    candidate = _candidate(
        Region(("left_matmul", "left_add"), ("x",), ("pre_relu",)),
        Region(("right_fused_linear",), ("x",), ("pre_relu",)),
    )
    result = StaticEquivalenceValidator().validate(left, right, candidate, {"x": "x"})
    assert result.status == "structural"
    assert result.rule_id is None


def test_gemm_with_wrong_beta_is_rejected(corpus_dir):
    left = GraphIR.from_path(str(corpus_dir / "invalid_gemm.left.onnx"))
    right = GraphIR.from_path(str(corpus_dir / "invalid_gemm.right.onnx"))
    candidate = _candidate(
        Region(("left_matmul", "left_add"), ("x",), ("pre_relu",)),
        Region(("right_gemm",), ("x",), ("pre_relu",)),
    )
    result = StaticEquivalenceValidator().validate(left, right, candidate, {"x": "x"})
    assert result.status == "rejected"
    assert result.rule_id == "matmul_add_to_gemm"


def test_schema_guard_rejects_operator_outside_declared_opset():
    # Declare opset 1 but use Gelu (added at opset 20); the checker is skipped
    # so the mismatch survives into the IR for the guard to catch.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    nodes = [
        helper.make_node("Gelu", ["x"], ["g"], name="gelu"),
        helper.make_node("Relu", ["g"], ["y"], name="relu"),
    ]
    graph = helper.make_graph(nodes, "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 1)])
    ir = GraphIR.from_model(model, infer_shapes=False)

    candidate = _candidate(Region(("gelu",), ("x",), ("g",)), Region((), ("x",), ("x",)))
    result = StaticEquivalenceValidator().validate(ir, ir, candidate, {"x": "x"})
    assert result.status == "rejected"
    assert any("unavailable in the declared opset" in detail for detail in result.details)


def test_gelu_schema_available_in_modern_opset():
    # Sanity: the same operator is fine once the opset is high enough.
    assert onnx.defs.get_schema("Gelu", 20, "") is not None
