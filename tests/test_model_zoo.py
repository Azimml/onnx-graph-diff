from __future__ import annotations

import json

import onnx

from onnx_graph_diff.evaluation import evaluate_corpus, summarize
from onnx_graph_diff.matcher import FingerprintMatcher
from onnx_graph_diff.model_zoo import generate_model_zoo


def test_generated_model_zoo_and_lineage(tmp_path):
    corpus = generate_model_zoo(tmp_path, seed=99)
    assert len(corpus) == 8
    assert {item["architecture"] for item in corpus} == {
        "mlp_unfused",
        "mlp_fused",
        "cnn",
        "residual_mlp",
        "transformer_like",
    }
    assert sum(len(item["regions"]) for item in corpus) == 22
    for item in corpus:
        onnx.checker.check_model(onnx.load(tmp_path / item["left"]))
        onnx.checker.check_model(onnx.load(tmp_path / item["right"]))
        assert all(region["left_anchor"] == region["right_anchor"] for region in item["regions"])
        assert all(region["applied_rule_id"] in item["rule_ids"] for region in item["regions"])
    saved = json.loads((tmp_path / "corpus.json").read_text(encoding="utf-8"))
    assert saved == corpus


def test_fingerprint_matches_all_generated_zoo_variants(tmp_path):
    generate_model_zoo(tmp_path, seed=100)
    records = evaluate_corpus(tmp_path, lambda _: FingerprintMatcher())
    summary = summarize(records)
    assert summary["outcome_accuracy"] == 1.0
    assert summary["anchor_accuracy"] == 1.0
    assert summary["region_accuracy"] == 1.0
    assert summary["validated_rule_accuracy"] == 1.0
    assert summary["proven_rate"] == 1.0
