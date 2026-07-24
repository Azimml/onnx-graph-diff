from __future__ import annotations

from onnx_graph_diff.fingerprint import COMMUTATIVE_OPS, FingerprintIndex
from onnx_graph_diff.ir import GraphIR


def _index(corpus_dir) -> tuple[GraphIR, FingerprintIndex]:
    graph = GraphIR.from_path(str(corpus_dir / "matmul_add_gemm.left.onnx"))
    return graph, FingerprintIndex(graph)


def test_fingerprints_are_deterministic(corpus_dir):
    graph, first = _index(corpus_dir)
    second = FingerprintIndex(graph)
    for node_id in graph.node_order:
        assert first[node_id].local == second[node_id].local
        assert first[node_id].forward == second[node_id].forward
        assert first[node_id].backward == second[node_id].backward


def test_output_distance_decreases_toward_graph_output(corpus_dir):
    graph, index = _index(corpus_dir)
    # matmul -> add -> relu, with relu feeding the graph output.
    distances = [index[node_id].output_distance for node_id in graph.node_order]
    assert distances == sorted(distances, reverse=True)
    assert min(distances) == 0  # the output-producing node is at distance 0


def test_distinct_ops_have_distinct_local_fingerprints(corpus_dir):
    graph, index = _index(corpus_dir)
    locals_ = {index[node_id].local for node_id in graph.node_order}
    assert len(locals_) == len(graph.node_order)


def test_downstream_context_hash_is_stable_and_depth_sensitive(corpus_dir):
    graph, index = _index(corpus_dir)
    root = graph.node_order[0]
    assert index.downstream_context_hash(root, depth=2) == index.downstream_context_hash(root, depth=2)
    # A deeper window sees more of the graph, so the digest changes with depth.
    assert index.downstream_context_hash(root, depth=0) != index.downstream_context_hash(root, depth=3)


def test_commutative_ops_are_recognized():
    assert {"Add", "Mul", "Max", "Min"} <= COMMUTATIVE_OPS
    assert "Sub" not in COMMUTATIVE_OPS
    assert "MatMul" not in COMMUTATIVE_OPS
