from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 20


def _initializer(name: str, value: np.ndarray | list[float] | float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def _make_model(
    name: str,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] | None = None,
    *,
    custom_domains: dict[str, int] | None = None,
) -> onnx.ModelProto:
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=initializers or [])
    imports = [helper.make_opsetid("", OPSET)]
    imports.extend(helper.make_opsetid(domain, version) for domain, version in (custom_domains or {}).items())
    model = helper.make_model(graph, producer_name="onnx-graph-diff", opset_imports=imports)
    onnx.checker.check_model(model)
    return model


def _fixture_matmul_add_gemm() -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, Any]]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    weight = _initializer("weight", np.arange(12, dtype=np.float32).reshape(4, 3) / 10)
    bias = _initializer("bias", [0.1, -0.2, 0.3])
    left = _make_model(
        "matmul_add_left",
        [
            helper.make_node("MatMul", ["x", "weight"], ["mm"], name="left_matmul"),
            helper.make_node("Add", ["mm", "bias"], ["pre_relu"], name="left_add"),
            helper.make_node("Relu", ["pre_relu"], ["y"], name="left_relu"),
        ],
        [x],
        [y],
        [weight, bias],
    )
    right = _make_model(
        "matmul_add_right",
        [
            helper.make_node("Gemm", ["x", "weight", "bias"], ["pre_relu"], name="right_gemm"),
            helper.make_node("Relu", ["pre_relu"], ["y"], name="right_relu"),
        ],
        [x],
        [y],
        [weight, bias],
    )
    return (
        left,
        right,
        _manifest(
            "matmul_add_to_gemm",
            ["left_matmul", "left_add"],
            ["right_gemm"],
            "left_relu",
            "right_relu",
            proven=True,
        ),
    )


def _fixture_identity_elimination() -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, Any]]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    left = _make_model(
        "identity_left",
        [
            helper.make_node("Identity", ["x"], ["identity_out"], name="left_identity"),
            helper.make_node("Relu", ["identity_out"], ["y"], name="left_relu"),
        ],
        [x],
        [y],
    )
    right = _make_model(
        "identity_right",
        [helper.make_node("Relu", ["x"], ["y"], name="right_relu")],
        [x],
        [y],
    )
    return (
        left,
        right,
        _manifest(
            "identity_elimination",
            ["left_identity"],
            [],
            "left_relu",
            "right_relu",
            proven=True,
        ),
    )


def _fixture_double_transpose() -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, Any]]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    perm = [0, 2, 1]
    left = _make_model(
        "double_transpose_left",
        [
            helper.make_node("Transpose", ["x"], ["t1"], name="left_transpose_1", perm=perm),
            helper.make_node("Transpose", ["t1"], ["t2"], name="left_transpose_2", perm=perm),
            helper.make_node("Relu", ["t2"], ["y"], name="left_relu"),
        ],
        [x],
        [y],
    )
    right = _make_model(
        "double_transpose_right",
        [helper.make_node("Relu", ["x"], ["y"], name="right_relu")],
        [x],
        [y],
    )
    return (
        left,
        right,
        _manifest(
            "double_transpose_elimination",
            ["left_transpose_1", "left_transpose_2"],
            [],
            "left_relu",
            "right_relu",
            proven=True,
        ),
    )


def _fixture_gelu_fusion() -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, Any]]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    constants = [
        _initializer("sqrt_two", math.sqrt(2.0)),
        _initializer("one", 1.0),
        _initializer("half", 0.5),
    ]
    left = _make_model(
        "gelu_left",
        [
            helper.make_node("Div", ["x", "sqrt_two"], ["div"], name="left_div"),
            helper.make_node("Erf", ["div"], ["erf"], name="left_erf"),
            helper.make_node("Add", ["erf", "one"], ["plus_one"], name="left_add"),
            helper.make_node("Mul", ["x", "plus_one"], ["scaled_x"], name="left_mul_x"),
            helper.make_node("Mul", ["scaled_x", "half"], ["gelu_out"], name="left_mul_half"),
            helper.make_node("Identity", ["gelu_out"], ["y"], name="left_anchor"),
        ],
        [x],
        [y],
        constants,
    )
    right = _make_model(
        "gelu_right",
        [
            helper.make_node("Gelu", ["x"], ["gelu_out"], name="right_gelu", approximate="none"),
            helper.make_node("Identity", ["gelu_out"], ["y"], name="right_anchor"),
        ],
        [x],
        [y],
    )
    return (
        left,
        right,
        _manifest(
            "erf_gelu_fusion",
            ["left_div", "left_erf", "left_add", "left_mul_x", "left_mul_half"],
            ["right_gelu"],
            "left_anchor",
            "right_anchor",
            proven=True,
        ),
    )


def _fixture_transpose_elementwise() -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, Any]]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    perm = [0, 2, 1]
    left = _make_model(
        "transpose_elementwise_left",
        [
            helper.make_node("Transpose", ["x"], ["t1"], name="left_transpose_in", perm=perm),
            helper.make_node("Relu", ["t1"], ["relu_t"], name="left_inner_relu"),
            helper.make_node("Transpose", ["relu_t"], ["restored"], name="left_transpose_out", perm=perm),
            helper.make_node("Identity", ["restored"], ["y"], name="left_anchor"),
        ],
        [x],
        [y],
    )
    right = _make_model(
        "transpose_elementwise_right",
        [
            helper.make_node("Relu", ["x"], ["restored"], name="right_relu"),
            helper.make_node("Identity", ["restored"], ["y"], name="right_anchor"),
        ],
        [x],
        [y],
    )
    return (
        left,
        right,
        _manifest(
            "transpose_elementwise_elimination",
            ["left_transpose_in", "left_inner_relu", "left_transpose_out"],
            ["right_relu"],
            "left_anchor",
            "right_anchor",
            proven=True,
        ),
    )


def _fixture_unknown_fused_linear() -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, Any]]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    weight = _initializer("weight", np.arange(12, dtype=np.float32).reshape(4, 3) / 10)
    bias = _initializer("bias", [0.1, -0.2, 0.3])
    left = _make_model(
        "unknown_fused_left",
        [
            helper.make_node("MatMul", ["x", "weight"], ["mm"], name="left_matmul"),
            helper.make_node("Add", ["mm", "bias"], ["pre_relu"], name="left_add"),
            helper.make_node("Relu", ["pre_relu"], ["y"], name="left_relu"),
        ],
        [x],
        [y],
        [weight, bias],
    )
    right = _make_model(
        "unknown_fused_right",
        [
            helper.make_node(
                "FusedLinear",
                ["x", "weight", "bias"],
                ["pre_relu"],
                name="right_fused_linear",
                domain="com.example.optim",
            ),
            helper.make_node("Relu", ["pre_relu"], ["y"], name="right_relu"),
        ],
        [x],
        [y],
        [weight, bias],
        custom_domains={"com.example.optim": 1},
    )
    return (
        left,
        right,
        _manifest(
            "unknown_fused_linear",
            ["left_matmul", "left_add"],
            ["right_fused_linear"],
            "left_relu",
            "right_relu",
            proven=False,
        ),
    )


def _fixture_invalid_gemm() -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, Any]]:
    left, _, _ = _fixture_matmul_add_gemm()
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    weight = _initializer("weight", np.arange(12, dtype=np.float32).reshape(4, 3) / 10)
    bias = _initializer("bias", [0.1, -0.2, 0.3])
    right = _make_model(
        "invalid_gemm_right",
        [
            helper.make_node("Gemm", ["x", "weight", "bias"], ["pre_relu"], name="right_gemm", beta=0.0),
            helper.make_node("Relu", ["pre_relu"], ["y"], name="right_relu"),
        ],
        [x],
        [y],
        [weight, bias],
    )
    return (
        left,
        right,
        _manifest(
            "not_equivalent",
            ["left_matmul", "left_add"],
            ["right_gemm"],
            "left_relu",
            "right_relu",
            proven=False,
            equivalent=False,
        ),
    )


def _fixture_residual_gemm() -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, Any]]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    weight = _initializer("weight", np.eye(4, dtype=np.float32) * 0.75)
    bias = _initializer("bias", [0.1, -0.2, 0.3, -0.4])
    left = _make_model(
        "residual_gemm_left",
        [
            helper.make_node("MatMul", ["x", "weight"], ["mm"], name="left_matmul"),
            helper.make_node("Add", ["mm", "bias"], ["main"], name="left_bias_add"),
            helper.make_node("Add", ["main", "x"], ["residual"], name="left_residual_add"),
            helper.make_node("Relu", ["residual"], ["y"], name="left_relu"),
        ],
        [x],
        [y],
        [weight, bias],
    )
    right = _make_model(
        "residual_gemm_right",
        [
            helper.make_node("Gemm", ["x", "weight", "bias"], ["main"], name="right_gemm"),
            helper.make_node("Add", ["main", "x"], ["residual"], name="right_residual_add"),
            helper.make_node("Relu", ["residual"], ["y"], name="right_relu"),
        ],
        [x],
        [y],
        [weight, bias],
    )
    return (
        left,
        right,
        _manifest(
            "matmul_add_to_gemm",
            ["left_matmul", "left_bias_add"],
            ["right_gemm"],
            "left_residual_add",
            "right_residual_add",
            proven=True,
        ),
    )


def _fixture_output_matmul_gemm() -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, Any]]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    weight = _initializer("weight", np.arange(12, dtype=np.float32).reshape(4, 3) / 10)
    bias = _initializer("bias", [0.1, -0.2, 0.3])
    left = _make_model(
        "output_matmul_left",
        [
            helper.make_node("MatMul", ["x", "weight"], ["mm"], name="output_matmul"),
            helper.make_node("Add", ["mm", "bias"], ["y"], name="output_add"),
        ],
        [x],
        [y],
        [weight, bias],
    )
    right = _make_model(
        "output_gemm_right",
        [helper.make_node("Gemm", ["x", "weight", "bias"], ["y"], name="output_gemm")],
        [x],
        [y],
        [weight, bias],
    )
    return (
        left,
        right,
        _manifest(
            "matmul_add_to_gemm",
            ["output_matmul", "output_add"],
            ["output_gemm"],
            "__graph_output__:0",
            "__graph_output__:0",
            proven=True,
        ),
    )


def _fixture_combined_pipeline() -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, Any]]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    weight = _initializer("weight", np.arange(12, dtype=np.float32).reshape(4, 3) / 10)
    bias = _initializer("bias", [0.1, -0.2, 0.3])
    perm = [1, 0]
    left = _make_model(
        "combined_left",
        [
            helper.make_node("Identity", ["x"], ["x_identity"], name="left_identity"),
            helper.make_node("Relu", ["x_identity"], ["stage1"], name="left_anchor_1"),
            helper.make_node("MatMul", ["stage1", "weight"], ["mm"], name="left_matmul"),
            helper.make_node("Add", ["mm", "bias"], ["linear"], name="left_bias_add"),
            helper.make_node("Relu", ["linear"], ["stage2"], name="left_anchor_2"),
            helper.make_node("Transpose", ["stage2"], ["t1"], name="left_transpose_1", perm=perm),
            helper.make_node("Transpose", ["t1"], ["stage3"], name="left_transpose_2", perm=perm),
            helper.make_node("Identity", ["stage3"], ["y"], name="left_anchor_3"),
        ],
        [x],
        [y],
        [weight, bias],
    )
    right = _make_model(
        "combined_right",
        [
            helper.make_node("Relu", ["x"], ["stage1"], name="right_anchor_1"),
            helper.make_node("Gemm", ["stage1", "weight", "bias"], ["linear"], name="right_gemm"),
            helper.make_node("Relu", ["linear"], ["stage2"], name="right_anchor_2"),
            helper.make_node("Identity", ["stage2"], ["y"], name="right_anchor_3"),
        ],
        [x],
        [y],
        [weight, bias],
    )
    regions = [
        {
            "rule_id": "identity_elimination",
            "left_region": ["left_identity"],
            "right_region": [],
            "left_anchor": "left_anchor_1",
            "right_anchor": "right_anchor_1",
            "proven": True,
        },
        {
            "rule_id": "matmul_add_to_gemm",
            "left_region": ["left_matmul", "left_bias_add"],
            "right_region": ["right_gemm"],
            "left_anchor": "left_anchor_2",
            "right_anchor": "right_anchor_2",
            "proven": True,
        },
        {
            "rule_id": "double_transpose_elimination",
            "left_region": ["left_transpose_1", "left_transpose_2"],
            "right_region": [],
            "left_anchor": "left_anchor_3",
            "right_anchor": "right_anchor_3",
            "proven": True,
        },
    ]
    return (
        left,
        right,
        {
            **_manifest(
                regions[0]["rule_id"],
                regions[0]["left_region"],
                regions[0]["right_region"],
                regions[0]["left_anchor"],
                regions[0]["right_anchor"],
                proven=True,
            ),
            "regions": regions,
        },
    )


def _manifest(
    rule_id: str,
    left_region: list[str],
    right_region: list[str],
    left_anchor: str,
    right_anchor: str,
    *,
    proven: bool,
    equivalent: bool = True,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "left_region": left_region,
        "right_region": right_region,
        "left_anchor": left_anchor,
        "right_anchor": right_anchor,
        "proven": proven,
        "equivalent": equivalent,
        "opset": OPSET,
    }


FIXTURE_BUILDERS = {
    "matmul_add_gemm": _fixture_matmul_add_gemm,
    "identity_elimination": _fixture_identity_elimination,
    "double_transpose": _fixture_double_transpose,
    "gelu_fusion": _fixture_gelu_fusion,
    "transpose_elementwise": _fixture_transpose_elementwise,
    "unknown_fused_linear": _fixture_unknown_fused_linear,
    "invalid_gemm": _fixture_invalid_gemm,
    "residual_gemm": _fixture_residual_gemm,
    "output_matmul_gemm": _fixture_output_matmul_gemm,
    "combined_pipeline": _fixture_combined_pipeline,
}


def generate_fixture_corpus(output_dir: str | Path) -> list[dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corpus: list[dict[str, Any]] = []
    for name, builder in FIXTURE_BUILDERS.items():
        left, right, manifest = builder()
        left_path = output / f"{name}.left.onnx"
        right_path = output / f"{name}.right.onnx"
        onnx.save(left, left_path)
        onnx.save(right, right_path)
        record = {
            "name": name,
            "left": left_path.name,
            "right": right_path.name,
            **manifest,
        }
        (output / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        corpus.append(record)
    (output / "corpus.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    return corpus
