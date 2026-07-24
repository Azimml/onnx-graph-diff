from __future__ import annotations

import copy

import numpy as np
import onnx
from onnx import numpy_helper

from onnx_graph_diff.ir import GraphIR
from onnx_graph_diff.matcher import FingerprintMatcher
from onnx_graph_diff.model_zoo import build_cnn
from onnx_graph_diff.rewrite import RuleEngine


def test_rewrite_engine_applies_fusion_and_decomposition(corpus_dir):
    engine = RuleEngine()
    cases = [
        ("identity_elimination.left.onnx", "identity_elimination", 1),
        ("double_transpose.left.onnx", "double_transpose_elimination", 1),
        ("matmul_add_gemm.left.onnx", "matmul_add_to_gemm", 1),
        ("gelu_fusion.left.onnx", "erf_gelu_fusion", 1),
        ("transpose_elementwise.left.onnx", "transpose_elementwise_elimination", 1),
        ("matmul_add_gemm.right.onnx", "gemm_decomposition", 1),
        ("gelu_fusion.right.onnx", "gelu_decomposition", 1),
    ]
    for filename, rule_id, count in cases:
        outcome = engine.apply(onnx.load(corpus_dir / filename), [rule_id])
        assert len(outcome.records) == count
        assert outcome.records[0].rule_id == rule_id
        onnx.checker.check_model(outcome.model)


def test_rewrite_engine_can_finish_at_virtual_graph_output(corpus_dir):
    outcome = RuleEngine().apply(
        onnx.load(corpus_dir / "output_matmul_gemm.left.onnx"),
        ["matmul_add_to_gemm"],
    )
    assert len(outcome.records) == 1
    assert outcome.records[0].left_anchor == "__graph_output__:0"
    assert outcome.records[0].right_anchor == "__graph_output__:0"
    assert [node.op_type for node in outcome.model.graph.node] == ["Gemm"]
    assert outcome.model.graph.output[0].name == "y"


def test_conv_batchnorm_fusion_is_proven_statically():
    baseline = build_cnn(123)
    outcome = RuleEngine().apply(baseline, ["conv_batchnorm_fusion"])
    assert len(outcome.records) == 1
    result = FingerprintMatcher().compare(GraphIR.from_model(baseline), GraphIR.from_model(outcome.model))
    region = next(event for event in result.events if event.kind == "region")
    assert region.proof.status == "proven"
    assert region.proof.rule_id == "conv_batchnorm_fusion"


def test_conv_batchnorm_certificate_rejects_modified_folded_weight():
    baseline = build_cnn(456)
    outcome = RuleEngine().apply(baseline, ["conv_batchnorm_fusion"])
    modified = copy.deepcopy(outcome.model)
    fused_conv = next(node for node in modified.graph.node if node.name.startswith("fused_conv_bn"))
    weight_name = fused_conv.input[1]
    tensor = next(item for item in modified.graph.initializer if item.name == weight_name)
    array = numpy_helper.to_array(tensor).copy()
    array.flat[0] += np.float32(0.25)
    replacement = numpy_helper.from_array(array, name=weight_name)
    index = next(index for index, item in enumerate(modified.graph.initializer) if item.name == weight_name)
    modified.graph.initializer[index].CopyFrom(replacement)

    matcher = FingerprintMatcher()
    left = GraphIR.from_model(baseline)
    right = GraphIR.from_model(modified)
    candidates = matcher.find_anchor_candidates(
        left,
        right,
        matcher._ready(left, set()),
        matcher._ready(right, set()),
        set(),
        set(),
        matcher._initial_input_mapping(left, right),
    )
    expected = next(
        candidate
        for candidate in candidates
        if candidate.left_anchor == "cnn_anchor_1" and candidate.right_anchor == "cnn_anchor_1"
    )
    proof = matcher.validator.validate(left, right, expected, {"x": "x"})
    assert proof.status == "rejected"
