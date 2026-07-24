from __future__ import annotations

from onnx_graph_diff.evaluation import score_result, summarize
from onnx_graph_diff.ir import GraphIR
from onnx_graph_diff.matcher import FingerprintMatcher


def test_fingerprint_corpus(corpus_dir, corpus):
    records = []
    for manifest in corpus:
        result = FingerprintMatcher().compare(
            GraphIR.from_path(str(corpus_dir / manifest["left"])),
            GraphIR.from_path(str(corpus_dir / manifest["right"])),
        )
        records.append(score_result(manifest, result))
        if manifest["name"] == "invalid_gemm":
            assert not result.completed
            assert not [event for event in result.events if event.kind == "region"]
        else:
            assert result.completed

    summary = summarize(records)
    assert summary["anchor_accuracy"] == 1.0
    assert summary["region_accuracy"] == 1.0
    assert summary["rule_accuracy"] == 1.0
    assert summary["false_positive_rate"] == 0.0


def test_combined_pipeline_has_three_minimal_regions(corpus_dir):
    result = FingerprintMatcher().compare(
        GraphIR.from_path(str(corpus_dir / "combined_pipeline.left.onnx")),
        GraphIR.from_path(str(corpus_dir / "combined_pipeline.right.onnx")),
    )
    regions = [event for event in result.events if event.kind == "region"]
    assert [event.proof.rule_id for event in regions] == [
        "identity_elimination",
        "matmul_add_to_gemm",
        "double_transpose_elimination",
    ]


def test_opaque_custom_operator_is_not_claimed_as_proven(corpus_dir):
    result = FingerprintMatcher().compare(
        GraphIR.from_path(str(corpus_dir / "unknown_fused_linear.left.onnx")),
        GraphIR.from_path(str(corpus_dir / "unknown_fused_linear.right.onnx")),
    )
    region = next(event for event in result.events if event.kind == "region")
    assert region.proof.status == "structural"
    assert region.proof.rule_id is None


def test_virtual_graph_output_anchor_completes_terminal_rewrite(corpus_dir):
    result = FingerprintMatcher().compare(
        GraphIR.from_path(str(corpus_dir / "output_matmul_gemm.left.onnx")),
        GraphIR.from_path(str(corpus_dir / "output_matmul_gemm.right.onnx")),
    )
    region = next(event for event in result.events if event.kind == "region")
    assert result.completed
    assert region.left_anchor == "__graph_output__:0"
    assert region.right_anchor == "__graph_output__:0"
    assert region.proof.status == "proven"
