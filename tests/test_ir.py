from __future__ import annotations

from onnx_graph_diff.ir import GraphIR


def _graph(corpus_dir) -> GraphIR:
    return GraphIR.from_path(str(corpus_dir / "matmul_add_gemm.left.onnx"))


def test_node_order_and_index_agree(corpus_dir):
    graph = _graph(corpus_dir)
    assert graph.node_order == ["left_matmul", "left_add", "left_relu"]
    for expected_index, node_id in enumerate(graph.node_order):
        assert graph.node_index(node_id) == expected_index


def test_parents_and_children_follow_edges(corpus_dir):
    graph = _graph(corpus_dir)
    assert graph.parents("left_matmul") == ()  # fed only by input/initializers
    assert graph.parents("left_add") == ("left_matmul",)
    assert graph.children("left_matmul") == ("left_add",)
    assert graph.children("left_relu") == ()  # produces the graph output only


def test_graph_inputs_and_outputs(corpus_dir):
    graph = _graph(corpus_dir)
    assert graph.graph_inputs == ["x"]
    assert graph.graph_outputs == ["y"]


def test_descendants_respects_hop_and_node_budgets(corpus_dir):
    graph = _graph(corpus_dir)
    assert graph.descendants(["left_matmul"], max_hops=1, max_nodes=10) == ["left_matmul", "left_add"]
    assert graph.descendants(["left_matmul"], max_hops=9, max_nodes=10) == graph.node_order
    # The node budget caps the window regardless of available depth.
    assert graph.descendants(["left_matmul"], max_hops=9, max_nodes=1) == ["left_matmul"]


def test_validate_opset_for_standard_and_custom_domains(corpus_dir):
    standard = _graph(corpus_dir)
    assert standard.validate_opset("left_matmul") is True

    custom = GraphIR.from_path(str(corpus_dir / "unknown_fused_linear.right.onnx"))
    # A custom-domain op is "valid" precisely because its domain was imported.
    assert custom.nodes["right_fused_linear"].domain == "com.example.optim"
    assert custom.validate_opset("right_fused_linear") is True
