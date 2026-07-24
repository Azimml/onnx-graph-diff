# Contributing to onnx-graph-diff

Thanks for your interest in improving `onnx-graph-diff`. This document describes
how to set up a development environment and the checks a change is expected to
pass.

## Development environment

Requires **Python 3.10+**.

```bash
git clone https://github.com/Azimml/onnx-graph-diff.git
cd onnx-graph-diff
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before you open a pull request

Run the same checks CI runs:

```bash
ruff check .          # lint
ruff format --check . # formatting
mypy                  # static type check (advisory)
pytest                # full offline test suite
```

The test suite runs entirely offline: the LLM routes are exercised through fake
clients, so no API key or network access is required.

## Guidelines

- **Keep the equivalence checker independent.** The matcher proposes regions; the
  checker in `equivalence.py` re-derives and validates them. Never let a matching
  route decide equivalence, and never claim `proven` for a rule whose side
  conditions have not been checked.
- **Add a certificate with a rule.** A new proven rewrite needs (1) a
  `RuleValidator` in `equivalence.py` that checks every side condition, (2) a
  labelled fixture in `fixtures.py`, and (3) a corpus assertion that both an
  equivalent and a deliberately-broken variant are classified correctly.
- **One logical change per commit** with a conventional-commit style message
  (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`).
- **Prefer deterministic tests.** Anything touching the fingerprint route must be
  reproducible; use the fixture corpus and the fake LLM clients for the assisted
  routes.

## Reporting bugs

Please open an issue with the two ONNX models (or a minimal reproducer that
generates them), the exact command line, and the JSON summary that was printed.
See the issue templates under `.github/ISSUE_TEMPLATE/`.
