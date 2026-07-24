from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

from onnx_graph_diff.cli import build_parser, main
from onnx_graph_diff.ir import GraphIR
from onnx_graph_diff.matcher import FingerprintMatcher
from onnx_graph_diff.report import build_report_data, render_html, render_markdown, write_comparison_report


class _Parser(HTMLParser):
    pass


def _report(corpus_dir: Path, name: str):
    left_path = corpus_dir / f"{name}.left.onnx"
    right_path = corpus_dir / f"{name}.right.onnx"
    left = GraphIR.from_path(str(left_path))
    right = GraphIR.from_path(str(right_path))
    result = FingerprintMatcher().compare(left, right)
    return build_report_data(
        left,
        right,
        result,
        left_path=str(left_path),
        right_path=str(right_path),
    )


def test_proven_report_contains_readable_rule_and_evidence(corpus_dir, tmp_path):
    data = _report(corpus_dir, "matmul_add_gemm")
    assert data["conclusion"]["code"] == "proven"
    assert data["regions"][0]["rule_id"] == "matmul_add_to_gemm"
    assert data["regions"][0]["status"] == "proven"
    assert data["exact_pairs"][0]["match_reason"]

    written = write_comparison_report(data, tmp_path / "comparison.html")
    html_text = written["html"].read_text(encoding="utf-8")
    json_data = json.loads(written["json"].read_text(encoding="utf-8"))
    _Parser().feed(html_text)
    assert "MatMul -&gt; Add" in html_text
    assert "matmul_add_to_gemm" in html_text
    assert "Exact matches, node by node" in html_text
    assert "Why it matched" in html_text
    assert json_data["conclusion"]["code"] == "proven"
    assert json_data["schema_version"] == 2


def test_report_distinguishes_structural_and_unresolved(corpus_dir):
    structural = _report(corpus_dir, "unknown_fused_linear")
    invalid = _report(corpus_dir, "invalid_gemm")
    assert structural["conclusion"]["code"] == "structural"
    assert structural["regions"][0]["status"] == "structural"
    assert invalid["conclusion"]["code"] == "unresolved"
    assert invalid["comparison"]["unmatched_left_count"] > 0
    assert invalid["comparison"]["unmatched_right_count"] > 0
    assert "Unresolved differences remain" in render_markdown(invalid)


def test_html_report_escapes_model_content(corpus_dir):
    data = _report(corpus_dir, "identity_elimination")
    data["left_model"]["name"] = '<script>alert("x")</script>'
    rendered = render_html(data)
    assert "<script>alert" not in rendered
    assert "&lt;script&gt;alert" in rendered


def test_public_cli_directly_compares_two_models(corpus_dir, tmp_path, capsys):
    output = tmp_path / "direct.html"
    exit_code = main(
        [
            str(corpus_dir / "combined_pipeline.left.onnx"),
            str(corpus_dir / "combined_pipeline.right.onnx"),
            "--report",
            str(output),
        ]
    )
    assert exit_code == 0
    assert output.exists()
    assert output.with_suffix(".json").exists()
    summary = json.loads(capsys.readouterr().out)
    assert summary["regions"] == 3
    assert summary["conclusion"]["code"] == "proven"

    help_text = build_parser().format_help()
    assert "left" in help_text and "right" in help_text
    assert "generate-fixtures" not in help_text
    assert "generate-model-zoo" not in help_text
