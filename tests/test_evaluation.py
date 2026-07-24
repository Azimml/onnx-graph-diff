from __future__ import annotations

from onnx_graph_diff.evaluation import score_result, summarize
from onnx_graph_diff.ir import GraphIR
from onnx_graph_diff.matcher import FingerprintMatcher


def test_summarize_of_empty_records_reports_zero_cases():
    assert summarize([]) == {"cases": 0}


def test_score_result_flags_proven_equivalent_rewrite(corpus_dir, corpus):
    manifest = next(item for item in corpus if item["name"] == "matmul_add_gemm")
    result = FingerprintMatcher().compare(
        GraphIR.from_path(str(corpus_dir / manifest["left"])),
        GraphIR.from_path(str(corpus_dir / manifest["right"])),
    )
    record = score_result(manifest, result)
    assert record["completed"] is True
    assert record["outcome_correct"] is True
    assert record["anchor_correct"] is True
    assert record["region_correct"] is True
    assert record["rule_correct"] is True
    assert record["false_positive"] is False


def test_score_result_does_not_flag_false_positive_on_inequivalent_pair(corpus_dir, corpus):
    manifest = next(item for item in corpus if item["name"] == "invalid_gemm")
    result = FingerprintMatcher().compare(
        GraphIR.from_path(str(corpus_dir / manifest["left"])),
        GraphIR.from_path(str(corpus_dir / manifest["right"])),
    )
    record = score_result(manifest, result)
    # The rewrite is not equivalent, and the matcher correctly refused to accept it.
    assert record["completed"] is False
    assert record["false_positive"] is False
    assert record["outcome_correct"] is True


def test_summarize_aggregates_rates_between_zero_and_one(corpus_dir, corpus):
    records = []
    for manifest in corpus:
        result = FingerprintMatcher().compare(
            GraphIR.from_path(str(corpus_dir / manifest["left"])),
            GraphIR.from_path(str(corpus_dir / manifest["right"])),
        )
        records.append(score_result(manifest, result))
    summary = summarize(records)
    assert summary["cases"] == len(corpus)
    for key in ("anchor_accuracy", "region_accuracy", "rule_accuracy", "false_positive_rate"):
        assert 0.0 <= summary[key] <= 1.0
