from __future__ import annotations

from onnx_graph_diff.context import ContextConfig, GraphContextBuilder
from onnx_graph_diff.ir import GraphIR
from onnx_graph_diff.model_zoo import build_cnn
from onnx_graph_diff.rewrite import RuleEngine


def test_required_parent_is_kept_even_when_budget_is_tiny(corpus_dir):
    graph = GraphIR.from_path(str(corpus_dir / "matmul_add_gemm.left.onnx"))
    context = GraphContextBuilder(
        ContextConfig(strategy="parents_only", max_hops=1, max_nodes_per_graph=1)
    )._build_side(graph, ["left_add"], {"left_matmul"})
    assert "left_matmul" in context["required_parent_nodes"]
    assert "left_matmul" in context["visible_node_ids"]
    assert "left_add" in context["visible_node_ids"]


def test_context_never_expands_initializer_payloads(corpus_dir):
    left = GraphIR.from_path(str(corpus_dir / "matmul_add_gemm.left.onnx"))
    right = GraphIR.from_path(str(corpus_dir / "matmul_add_gemm.right.onnx"))
    context = GraphContextBuilder().build(left, right, ["left_matmul"], ["right_gemm"], set(), set(), {"x": "x"})
    rendered = str(context)
    assert "0.100000" not in rendered
    assert "sha256" not in rendered
    assert "hash" in rendered
    assert context["constraints"]["required_parent_nodes_are_already_matched"] is True
    assert any(item["rule_id"] == "matmul_add_to_gemm" for item in context["rule_catalog"])


def test_coarse_anchor_pairs_are_sorted_by_frontier_distance(corpus_dir):
    left = GraphIR.from_path(str(corpus_dir / "combined_pipeline.left.onnx"))
    right = GraphIR.from_path(str(corpus_dir / "combined_pipeline.right.onnx"))
    context = GraphContextBuilder(ContextConfig(max_hops=8)).build(
        left,
        right,
        ["left_identity"],
        ["right_anchor_1"],
        set(),
        set(),
        {"x": "x"},
    )
    first = context["coarse_anchor_pairs"][0]
    assert (first["left"], first["right"]) == ("left_anchor_1", "right_anchor_1")


def test_coarse_pair_carries_deterministic_closed_regions():
    baseline = build_cnn(321)
    optimized = (
        RuleEngine()
        .apply(
            baseline,
            ["conv_batchnorm_fusion", "transpose_elementwise_elimination", "identity_elimination"],
        )
        .model
    )
    left = GraphIR.from_model(baseline)
    right = GraphIR.from_model(optimized)
    context = GraphContextBuilder(ContextConfig(max_hops=8)).build(
        left,
        right,
        ["cnn_transpose_in"],
        ["layout_relu__cnn_inner_relu"],
        {"cnn_conv", "cnn_batchnorm", "cnn_anchor_1"},
        {"fused_conv_bn__cnn_conv", "cnn_anchor_1"},
        {"x": "x", "cnn_bn_out": "cnn_bn_out", "cnn_stage1": "cnn_stage1"},
    )
    pair = next(
        item
        for item in context["coarse_anchor_pairs"]
        if item["left"] == "cnn_anchor_2" and item["right"] == "cnn_anchor_2"
    )
    assert pair["left_region_nodes"] == ["cnn_transpose_in", "cnn_inner_relu", "cnn_transpose_out"]
    assert pair["right_region_nodes"] == ["layout_relu__cnn_inner_relu"]
