from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper

from .ir import GraphIR
from .rewrite import RewriteRecord, RuleEngine

OPSET = 20


def _initializer(name: str, value: np.ndarray | list[float] | float | list[int]) -> onnx.TensorProto:
    array = np.asarray(value)
    if array.dtype.kind == "f":
        array = array.astype(np.float32)
    return numpy_helper.from_array(array, name=name)


def _model(
    name: str,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto],
) -> ModelProto:
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=initializers)
    model = helper.make_model(
        graph,
        producer_name="onnx-graph-diff-model-zoo",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def _erf_gelu_nodes(prefix: str, x: str, output: str) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    sqrt_name = f"{prefix}_sqrt_two"
    one_name = f"{prefix}_one"
    half_name = f"{prefix}_half"
    nodes = [
        helper.make_node("Div", [x, sqrt_name], [f"{prefix}_div"], name=f"{prefix}_div_node"),
        helper.make_node("Erf", [f"{prefix}_div"], [f"{prefix}_erf"], name=f"{prefix}_erf_node"),
        helper.make_node("Add", [f"{prefix}_erf", one_name], [f"{prefix}_plus_one"], name=f"{prefix}_add_node"),
        helper.make_node("Mul", [x, f"{prefix}_plus_one"], [f"{prefix}_mul_x"], name=f"{prefix}_mul_x_node"),
        helper.make_node("Mul", [f"{prefix}_mul_x", half_name], [output], name=f"{prefix}_mul_half_node"),
    ]
    constants = [
        _initializer(sqrt_name, math.sqrt(2.0)),
        _initializer(one_name, 1.0),
        _initializer(half_name, 0.5),
    ]
    return nodes, constants


def build_mlp_unfused(seed: int) -> ModelProto:
    rng = np.random.default_rng(seed)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 8])
    gelu_nodes, gelu_constants = _erf_gelu_nodes("mlp_gelu", "mlp_stage2", "mlp_gelu_out")
    initializers = [
        _initializer("mlp_w1", rng.normal(0, 0.2, (8, 16))),
        _initializer("mlp_b1", rng.normal(0, 0.1, (16,))),
        _initializer("mlp_w2", rng.normal(0, 0.2, (16, 8))),
        _initializer("mlp_b2", rng.normal(0, 0.1, (8,))),
        *gelu_constants,
    ]
    nodes = [
        helper.make_node("Identity", ["x"], ["mlp_x_identity"], name="mlp_identity"),
        helper.make_node("Relu", ["mlp_x_identity"], ["mlp_stage1"], name="mlp_anchor_1"),
        helper.make_node("MatMul", ["mlp_stage1", "mlp_w1"], ["mlp_mm1"], name="mlp_matmul_1"),
        helper.make_node("Add", ["mlp_mm1", "mlp_b1"], ["mlp_linear1"], name="mlp_add_1"),
        helper.make_node("Relu", ["mlp_linear1"], ["mlp_stage2"], name="mlp_anchor_2"),
        *gelu_nodes,
        helper.make_node("Tanh", ["mlp_gelu_out"], ["mlp_stage3"], name="mlp_anchor_3"),
        helper.make_node("MatMul", ["mlp_stage3", "mlp_w2"], ["mlp_mm2"], name="mlp_matmul_2"),
        helper.make_node("Add", ["mlp_mm2", "mlp_b2"], ["mlp_linear2"], name="mlp_add_2"),
        helper.make_node("Relu", ["mlp_linear2"], ["mlp_stage4"], name="mlp_anchor_4"),
        helper.make_node("Transpose", ["mlp_stage4"], ["mlp_t1"], name="mlp_transpose_1", perm=[1, 0]),
        helper.make_node("Transpose", ["mlp_t1"], ["mlp_t2"], name="mlp_transpose_2", perm=[1, 0]),
        helper.make_node("Sigmoid", ["mlp_t2"], ["y"], name="mlp_anchor_5"),
    ]
    return _model("seeded_mlp_unfused", nodes, [x], [y], initializers)


def build_mlp_fused(seed: int) -> ModelProto:
    rng = np.random.default_rng(seed + 11)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 8])
    initializers = [
        _initializer("fused_w1", rng.normal(0, 0.2, (8, 16))),
        _initializer("fused_b1", rng.normal(0, 0.1, (16,))),
        _initializer("fused_w2", rng.normal(0, 0.2, (16, 8))),
        _initializer("fused_b2", rng.normal(0, 0.1, (8,))),
    ]
    nodes = [
        helper.make_node("Gemm", ["x", "fused_w1", "fused_b1"], ["fused_linear1"], name="fused_gemm_1"),
        helper.make_node("Relu", ["fused_linear1"], ["fused_stage1"], name="fused_anchor_1"),
        helper.make_node("Gelu", ["fused_stage1"], ["fused_gelu_out"], name="fused_gelu", approximate="none"),
        helper.make_node("Tanh", ["fused_gelu_out"], ["fused_stage2"], name="fused_anchor_2"),
        helper.make_node("Gemm", ["fused_stage2", "fused_w2", "fused_b2"], ["fused_linear2"], name="fused_gemm_2"),
        helper.make_node("Sigmoid", ["fused_linear2"], ["y"], name="fused_anchor_3"),
    ]
    return _model("seeded_mlp_fused", nodes, [x], [y], initializers)


def build_cnn(seed: int) -> ModelProto:
    rng = np.random.default_rng(seed + 23)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])
    initializers = [
        _initializer("cnn_weight", rng.normal(0, 0.1, (4, 3, 3, 3))),
        _initializer("cnn_bias", rng.normal(0, 0.05, (4,))),
        _initializer("cnn_bn_scale", rng.normal(1.0, 0.05, (4,))),
        _initializer("cnn_bn_bias", rng.normal(0, 0.05, (4,))),
        _initializer("cnn_bn_mean", rng.normal(0, 0.1, (4,))),
        _initializer("cnn_bn_var", rng.uniform(0.5, 1.5, (4,))),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            ["x", "cnn_weight", "cnn_bias"],
            ["cnn_conv_out"],
            name="cnn_conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            "BatchNormalization",
            ["cnn_conv_out", "cnn_bn_scale", "cnn_bn_bias", "cnn_bn_mean", "cnn_bn_var"],
            ["cnn_bn_out"],
            name="cnn_batchnorm",
            epsilon=1e-5,
            training_mode=0,
        ),
        helper.make_node("Relu", ["cnn_bn_out"], ["cnn_stage1"], name="cnn_anchor_1"),
        helper.make_node("Transpose", ["cnn_stage1"], ["cnn_nhwc"], name="cnn_transpose_in", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["cnn_nhwc"], ["cnn_nhwc_relu"], name="cnn_inner_relu"),
        helper.make_node("Transpose", ["cnn_nhwc_relu"], ["cnn_stage2"], name="cnn_transpose_out", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["cnn_stage2"], ["cnn_stage3"], name="cnn_anchor_2"),
        helper.make_node("Identity", ["cnn_stage3"], ["cnn_identity_out"], name="cnn_identity"),
        helper.make_node("LeakyRelu", ["cnn_identity_out"], ["y"], name="cnn_anchor_3", alpha=0.1),
    ]
    return _model("seeded_cnn", nodes, [x], [y], initializers)


def build_residual_mlp(seed: int) -> ModelProto:
    rng = np.random.default_rng(seed + 37)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 8])
    initializers = [
        _initializer("res_w1", rng.normal(0, 0.15, (8, 8))),
        _initializer("res_b1", rng.normal(0, 0.05, (8,))),
        _initializer("res_w2", rng.normal(0, 0.15, (8, 8))),
        _initializer("res_b2", rng.normal(0, 0.05, (8,))),
    ]
    nodes = [
        helper.make_node("MatMul", ["x", "res_w1"], ["res_mm1"], name="res_matmul_1"),
        helper.make_node("Add", ["res_mm1", "res_b1"], ["res_main1"], name="res_add_1"),
        helper.make_node("Add", ["res_main1", "x"], ["residual1"], name="residual_anchor_1"),
        helper.make_node("Relu", ["residual1"], ["res_stage1"], name="residual_anchor_2"),
        helper.make_node("MatMul", ["res_stage1", "res_w2"], ["res_mm2"], name="res_matmul_2"),
        helper.make_node("Add", ["res_mm2", "res_b2"], ["res_main2"], name="res_add_2"),
        helper.make_node("Add", ["res_main2", "res_stage1"], ["residual2"], name="residual_anchor_3"),
        helper.make_node("Sigmoid", ["residual2"], ["y"], name="residual_anchor_4"),
    ]
    return _model("seeded_residual_mlp", nodes, [x], [y], initializers)


def build_transformer_like(seed: int) -> ModelProto:
    rng = np.random.default_rng(seed + 51)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4, 8])
    gelu_nodes, gelu_constants = _erf_gelu_nodes("tx_gelu", "tx_stage1", "tx_gelu_out")
    initializers = [
        _initializer("tx_flat_shape", np.asarray([8, 8], dtype=np.int64)),
        _initializer("tx_restore_shape", np.asarray([2, 4, 8], dtype=np.int64)),
        _initializer("tx_w1", rng.normal(0, 0.12, (8, 8))),
        _initializer("tx_b1", rng.normal(0, 0.04, (8,))),
        _initializer("tx_w2", rng.normal(0, 0.12, (8, 8))),
        _initializer("tx_b2", rng.normal(0, 0.04, (8,))),
        _initializer("tx_ln_scale", np.ones((8,), dtype=np.float32)),
        _initializer("tx_ln_bias", np.zeros((8,), dtype=np.float32)),
        *gelu_constants,
    ]
    nodes = [
        helper.make_node("Reshape", ["x", "tx_flat_shape"], ["tx_flat"], name="tx_reshape_in"),
        helper.make_node("MatMul", ["tx_flat", "tx_w1"], ["tx_mm1"], name="tx_matmul_1"),
        helper.make_node("Add", ["tx_mm1", "tx_b1"], ["tx_linear1"], name="tx_add_1"),
        helper.make_node("Relu", ["tx_linear1"], ["tx_stage1"], name="tx_anchor_1"),
        *gelu_nodes,
        helper.make_node("Tanh", ["tx_gelu_out"], ["tx_stage2"], name="tx_anchor_2"),
        helper.make_node("MatMul", ["tx_stage2", "tx_w2"], ["tx_mm2"], name="tx_matmul_2"),
        helper.make_node("Add", ["tx_mm2", "tx_b2"], ["tx_linear2"], name="tx_add_2"),
        helper.make_node("Relu", ["tx_linear2"], ["tx_stage3"], name="tx_anchor_3"),
        helper.make_node("Reshape", ["tx_stage3", "tx_restore_shape"], ["tx_restored"], name="tx_reshape_out"),
        helper.make_node("Add", ["tx_restored", "x"], ["tx_residual"], name="tx_residual_add"),
        helper.make_node(
            "LayerNormalization",
            ["tx_residual", "tx_ln_scale", "tx_ln_bias"],
            ["tx_norm"],
            name="tx_layernorm",
            axis=-1,
            epsilon=1e-5,
        ),
        helper.make_node("Sigmoid", ["tx_norm"], ["y"], name="tx_anchor_4"),
    ]
    return _model("seeded_transformer_like", nodes, [x], [y], initializers)


@dataclass(frozen=True)
class ZooSpec:
    name: str
    builder: Callable[[int], ModelProto]
    variants: dict[str, tuple[str, ...]]


SPECS = [
    ZooSpec(
        "mlp_unfused",
        build_mlp_unfused,
        {
            "linear_fusions": ("matmul_add_to_gemm",),
            "activation_layout": ("identity_elimination", "erf_gelu_fusion", "double_transpose_elimination"),
            "all_fusions": (
                "identity_elimination",
                "matmul_add_to_gemm",
                "erf_gelu_fusion",
                "double_transpose_elimination",
            ),
        },
    ),
    ZooSpec("mlp_fused", build_mlp_fused, {"decomposed": ("gemm_decomposition", "gelu_decomposition")}),
    ZooSpec(
        "cnn",
        build_cnn,
        {
            "conv_bn": ("conv_batchnorm_fusion",),
            "backend_all": ("conv_batchnorm_fusion", "transpose_elementwise_elimination", "identity_elimination"),
        },
    ),
    ZooSpec("residual_mlp", build_residual_mlp, {"linear_fusions": ("matmul_add_to_gemm",)}),
    ZooSpec(
        "transformer_like",
        build_transformer_like,
        {"fusions": ("matmul_add_to_gemm", "erf_gelu_fusion")},
    ),
]


VALIDATED_RULE_ID = {
    "gemm_decomposition": "matmul_add_to_gemm",
    "gelu_decomposition": "erf_gelu_fusion",
}


def _record_to_region(record: RewriteRecord) -> dict:
    return {
        "rule_id": VALIDATED_RULE_ID.get(record.rule_id, record.rule_id),
        "applied_rule_id": record.rule_id,
        "left_region": record.before_nodes,
        "right_region": record.after_nodes,
        "left_anchor": record.left_anchor,
        "right_anchor": record.right_anchor,
        "left_exit_values": record.left_exit_values,
        "right_exit_values": record.right_exit_values,
        "proven": True,
        "metadata": record.metadata,
    }


def generate_model_zoo(output_dir: str | Path, *, seed: int = 20260720) -> list[dict]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    engine = RuleEngine()
    corpus: list[dict] = []
    for spec_index, spec in enumerate(SPECS):
        baseline = spec.builder(seed + spec_index * 101)
        baseline_path = output / f"{spec.name}.baseline.onnx"
        onnx.save(baseline, baseline_path)
        baseline_ir = GraphIR.from_model(baseline)
        for variant_name, rule_ids in spec.variants.items():
            outcome = engine.apply(baseline, list(rule_ids))
            if not outcome.records:
                raise RuntimeError(f"No rules applied for {spec.name}:{variant_name}")
            variant_path = output / f"{spec.name}.{variant_name}.onnx"
            onnx.save(outcome.model, variant_path)
            regions = [_record_to_region(record) for record in outcome.records]
            regions.sort(
                key=lambda region: (
                    baseline_ir.node_index(region["left_anchor"])
                    if region["left_anchor"] in baseline_ir.nodes
                    else 1_000_000
                )
            )
            record = {
                "name": f"{spec.name}__{variant_name}",
                "architecture": spec.name,
                "variant": variant_name,
                "seed": seed + spec_index * 101,
                "left": baseline_path.name,
                "right": variant_path.name,
                "rule_ids": list(rule_ids),
                "regions": regions,
                "rule_id": regions[0]["rule_id"],
                "left_region": regions[0]["left_region"],
                "right_region": regions[0]["right_region"],
                "left_anchor": regions[0]["left_anchor"],
                "right_anchor": regions[0]["right_anchor"],
                "proven": True,
                "equivalent": True,
                "opset": OPSET,
            }
            (output / f"{record['name']}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            corpus.append(record)
    (output / "corpus.json").write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    return corpus
