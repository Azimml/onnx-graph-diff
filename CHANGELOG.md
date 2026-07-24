# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Module-level docstrings across the package documenting the IR, fingerprinting,
  matcher, equivalence checker, context builder, workflow, and reporting layers.
- Public API re-exports and `__version__` on the package root.
- `.editorconfig` and `.gitattributes` for consistent formatting and line endings.
- Developer tooling: `CONTRIBUTING.md`, a `Makefile` with common dev targets,
  and GitHub issue/PR templates.
- A runnable `examples/` script that compares two ONNX models end to end.
- Additional unit tests for fingerprint determinism, IR traversal, the
  equivalence certificates, evaluation scoring, provider key resolution, and
  context shape compatibility.

## [0.1.0] - 2026

### Added
- Initial public release.
- Three matching routes: deterministic fingerprinting, LLM-assisted anchor
  selection, and LLM-led agentic search.
- Independent, opset-aware static equivalence checker with certificates for
  `matmul_add_to_gemm`, `identity_elimination`, `double_transpose_elimination`,
  `transpose_elementwise_elimination`, `erf_gelu_fusion`, and
  `conv_batchnorm_fusion`.
- Pluggable OpenAI-compatible provider registry (DashScope/Qwen, OpenAI, and a
  generic OpenAI-compatible backend).
- HTML, Markdown, and JSON reports with a machine-readable JSON sidecar.

[Unreleased]: https://github.com/Azimml/onnx-graph-diff/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Azimml/onnx-graph-diff/releases/tag/v0.1.0
