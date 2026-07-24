#!/usr/bin/env python3
"""Minimal end-to-end example: generate a fixture pair and statically diff it.

This script needs no API key and no network access -- it uses the deterministic
fingerprint route on a bundled fixture (a ``MatMul + Add`` fused into a ``Gemm``).

Run it from a checkout after ``pip install -e ".[dev]"``::

    python examples/run_diff.py

It writes an HTML report plus a JSON sidecar into a temporary directory and
prints the comparison conclusion and each detected difference region.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from onnx_graph_diff.fixtures import generate_fixture_corpus
from onnx_graph_diff.ir import GraphIR
from onnx_graph_diff.matcher import FingerprintMatcher
from onnx_graph_diff.report import build_report_data, write_comparison_report


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="onnx-graph-diff-example-"))
    generate_fixture_corpus(workdir)

    left_path = workdir / "matmul_add_gemm.left.onnx"
    right_path = workdir / "matmul_add_gemm.right.onnx"

    left = GraphIR.from_path(str(left_path))
    right = GraphIR.from_path(str(right_path))

    result = FingerprintMatcher().compare(left, right)

    report_data = build_report_data(
        left,
        right,
        result,
        left_path=str(left_path),
        right_path=str(right_path),
    )
    written = write_comparison_report(
        report_data,
        str(workdir / "report.html"),
        report_format="html",
        json_sidecar=True,
    )

    conclusion = report_data["conclusion"]
    print(f"Conclusion: {conclusion['label']} ({conclusion['code']})")
    print(f"Completed:  {result.completed}")

    regions = [event for event in result.events if event.kind == "region"]
    print(f"Difference regions: {len(regions)}")
    for index, event in enumerate(regions, start=1):
        proof = event.proof
        left_ops = " -> ".join(left.nodes[node_id].op_type for node_id in event.left_nodes)
        right_ops = " -> ".join(right.nodes[node_id].op_type for node_id in event.right_nodes)
        status = proof.status if proof else "unknown"
        rule = proof.rule_id if proof else None
        print(f"  region #{index}: {left_ops}  <=>  {right_ops}  [{status}, rule={rule}]")

    for name, path in written.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
