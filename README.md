# onnx-graph-diff

[![CI](https://github.com/Azimml/onnx-graph-diff/actions/workflows/ci.yml/badge.svg)](https://github.com/Azimml/onnx-graph-diff/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Statically compare two ONNX models and explain the N:M subgraph differences that a graph-optimization pass introduced — without ever running the models.**

When a compiler or optimizer rewrites an ONNX graph (fusing `MatMul + Add` into `Gemm`, folding `Conv + BatchNormalization`, eliminating `Identity` nodes, collapsing double transposes, fusing an Erf-based GELU, ...), the optimized graph is no longer node-for-node comparable with the original. A naive textual or node-count diff produces noise. `onnx-graph-diff` instead finds where the two graphs *rejoin* after each local transformation, isolates the smallest mismatch region on each side, and then asks an **independent equivalence checker** whether that N:M rewrite is actually mathematically sound under the declared ONNX opset — or merely structurally plausible.

The result is a readable HTML / Markdown / JSON report that says, region by region: *this is a proven optimization*, *this is a structural rejoin we could not prove*, or *this candidate was rejected / left unresolved*.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [The three matching routes](#the-three-matching-routes)
- [The independent equivalence checker](#the-independent-equivalence-checker)
- [Install](#install)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [LLM providers](#llm-providers)
- [Sample report](#sample-report)
- [Architecture overview](#architecture-overview)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Why this exists

Graph-level optimizations change *structure* while intending to preserve *semantics*. Verifying that intent is hard:

- Running both models and comparing outputs is a **dynamic** check — it needs representative inputs, it is sensitive to floating-point tolerance, and it tells you *that* something differs, not *which rewrite* caused it or *why* it is safe.
- A plain graph diff drowns you in incidental renaming and reordering.

`onnx-graph-diff` is a **static** tool. It aligns the two DAGs, attributes each divergence to a concrete local rewrite, and validates that rewrite against ONNX operator semantics. Nothing is executed; no sample data is required.

## The three matching routes

All three routes share the same deterministic mechanics (topological readiness, exact-match commitment, region closure) and the same equivalence checker. They differ only in **how the rejoin anchors after a mismatch are chosen**.

| Route | CLI | How anchors are chosen | LLM calls |
| --- | --- | --- | --- |
| **Deterministic fingerprinting** | `--method fingerprint` | Structural fingerprints enumerate candidate rejoin anchors and pick the closest closed region. Fully reproducible. | none |
| **LLM-assisted anchor selection** | `--method llm` | The tool computes coarse anchor candidates deterministically, then the LLM selects among them and names the rule. Every choice is re-validated by the checker. | one per mismatch (with bounded retries) |
| **LLM-led search** | `--method llm-agent` | No candidates are supplied. The LLM drives the search over a windowed view of the graph, deciding `match` / `expand` / `unresolved`, while the tool enforces graph state, region closure, and equivalence. | several per mismatch (agentic loop with expansion) |

The deterministic route is the default and needs no API key. The two LLM routes are useful for the harder, unseen rewrites where fingerprints alone cannot decide, and for producing natural-language explanations — but **the LLM is never trusted to declare equivalence**. It only proposes; the checker disposes.

## The independent equivalence checker

The equivalence checker (`equivalence.py`) is deliberately **independent of the matcher**. Whatever route proposed a region, the checker re-derives the region from the graph itself and applies opset-aware rules. It never takes the proposer's word for the rule or the equivalence.

Each region is assigned one of four proof statuses:

- **`proven`** — a supported ONNX algebraic identity holds under the declared opset (with all its side conditions checked: matching initializers, attribute constraints, arity, dtype/shape compatibility).
- **`structural`** — the two sides rejoin cleanly and the shapes line up, but no supported certificate applies (e.g. an opaque custom-domain fused op). The rewrite is reported as *structurally rematched, not proven*.
- **`rejected`** — a known rule pattern matched but one of its conditions is violated, so equivalence is explicitly denied.
- **`unresolved`** — traversal could not close the region at all; nodes remain unmatched and need a new rule or manual review.

Built-in proven rules today:

| Rule id | Rewrite it validates |
| --- | --- |
| `matmul_add_to_gemm` | `MatMul → Add`  ⇔  `Gemm` |
| `identity_elimination` | `Identity` removed |
| `double_transpose_elimination` | `Transpose → Transpose` (inverse perms) removed |
| `transpose_elementwise_elimination` | `Transpose → elementwise → Transpose`  ⇔  elementwise |
| `erf_gelu_fusion` | `Div → Erf → Add → Mul → Mul`  ⇔  `Gelu` |
| `conv_batchnorm_fusion` | `Conv → BatchNormalization`  ⇔  folded `Conv` |

A schema guard first rejects any region whose standard-domain operators are unavailable in the declared opset, so proofs are never claimed outside the version they hold in.

## Install

Requires **Python 3.10+**.

```bash
git clone https://github.com/Azimml/onnx-graph-diff.git
cd onnx-graph-diff
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # drop [dev] for a runtime-only install
```

This installs the `onnx-graph-diff` console entry point.

## Quick start

Compare two models with the deterministic route (no API key needed):

```bash
onnx-graph-diff baseline.onnx optimized.onnx --method fingerprint
```

This writes `reports/baseline__vs__optimized.html` plus a machine-readable JSON sidecar, and prints a summary:

```json
{
  "conclusion": {
    "code": "proven",
    "label": "Optimization differences passed static validation",
    "detail": "Every detected difference region passed a mathematical-equivalence certificate under the currently supported ONNX opset rules."
  },
  "completed": true,
  "regions": 1,
  "unmatched_left": 0,
  "unmatched_right": 0
}
```

Use an LLM route for harder graphs (see [LLM providers](#llm-providers) for keys):

```bash
# LLM chooses among deterministically-generated anchor candidates
onnx-graph-diff baseline.onnx optimized.onnx --method llm --provider openai

# Fully LLM-led anchor search
onnx-graph-diff baseline.onnx optimized.onnx --method llm-agent --provider openai
```

## CLI reference

```
onnx-graph-diff LEFT RIGHT [options]
```

| Argument / flag | Default | Description |
| --- | --- | --- |
| `LEFT` | — | Path to the baseline / unoptimized ONNX model. |
| `RIGHT` | — | Path to the candidate / optimized ONNX model. |
| `--method {fingerprint,llm,llm-agent}` | `fingerprint` | Matching route (see above). |
| `--report PATH` | `reports/<left>__vs__<right>.<ext>` | Report output path. |
| `--report-format {html,markdown,json,all}` | `html` | Primary report format. HTML is standalone. |
| `--json-sidecar / --no-json-sidecar` | on | Also write the full machine-readable JSON next to HTML/Markdown. |
| `--explain / --no-explain` | on for LLM methods | Ask the LLM for a natural-language post-match explanation. |
| `--reject-structural-unknown` | off | Leave unknown transformations `unresolved` instead of accepting structural-only rejoins. |

**LLM matching and explanation** (used by `--method llm`/`llm-agent` and `--explain`):

| Flag | Default | Description |
| --- | --- | --- |
| `--provider {dashscope,qwen,openai,openai-compatible}` | `LLM_PROVIDER` env, else `dashscope` | LLM backend. See [LLM providers](#llm-providers). |
| `--model MODEL` | provider default | Chat model name. |
| `--base-url URL` | provider default | OpenAI-compatible base URL. |
| `--api-key-env NAME` | provider default | Environment variable to read the API key from. |
| `--thinking` | off | Enable the provider "thinking" stream (DashScope/Qwen only). |
| `--temperature FLOAT` | `0.0` | Sampling temperature. |
| `--seed INT` | `42` | Seed for reproducibility (where supported). |
| `--timeout FLOAT` | `120.0` | Per-request timeout in seconds. |
| `--context-strategy {parents_only,bounded,path_summary,full_window}` | `path_summary` | How much graph context the `llm` route serializes. |
| `--prompt-variant {anchor_v1,proof_v2,region_v3,region_v4}` | `region_v4` | Prompt used by `--method llm`. |
| `--agent-prompt {agent_v1..agent_v13}` | `agent_v13` | Prompt used by `--method llm-agent`. |
| `--agent-context-strategy {bounded,path_summary,full_window}` | `bounded` | Context strategy for the agent route. |
| `--max-hops`, `--max-nodes`, `--max-chars` | `8`, `20`, `24000` | Windowing / serialization budgets. |
| `--max-retries` | `1` | Retries after a rejected/invalid proposal. |
| `--no-coarse-candidates` | off | Withhold deterministic anchor candidates from the `llm` route. |
| `--agent-initial-hops`, `--agent-initial-nodes` | `2`, `8` | Starting agent window size. |
| `--agent-max-expansions`, `--agent-max-calls` | `3`, `8` | Agent search budgets per mismatch. |

Run `onnx-graph-diff --help` for the authoritative, always-current list.

## LLM providers

The LLM backend is **pluggable**. Provider specifics — default model, base URL, and which environment variable holds the API key — live in `providers.py` behind a `ProviderProfile`, and the client (`LLMClient`) is provider-agnostic. Select a provider with `--provider`, the `LLM_PROVIDER` environment variable, or the `provider=` field of `LLMConfig`.

| Provider | Endpoint | API-key env | Notes |
| --- | --- | --- | --- |
| `dashscope` (default) / `qwen` | DashScope OpenAI-compatible endpoint | `DASHSCOPE_API_KEY` | Original backend; supports the `--thinking` stream. |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | Public OpenAI API. |
| `openai-compatible` | your `--base-url` | `--api-key-env` (falls back to `LLM_API_KEY`, then `OPENAI_API_KEY`) | Any OpenAI-style server: vLLM, Together, Groq, Ollama, a self-hosted gateway, ... Requires `--model`. |

Examples:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
onnx-graph-diff a.onnx b.onnx --method llm --provider openai --model gpt-4o-mini

# DashScope / Qwen (backward compatible — the historical default)
export DASHSCOPE_API_KEY=...
onnx-graph-diff a.onnx b.onnx --method llm-agent

# Any OpenAI-compatible endpoint (e.g. a local vLLM or Ollama server)
export MY_GATEWAY_KEY=...
onnx-graph-diff a.onnx b.onnx --method llm \
  --provider openai-compatible \
  --base-url http://localhost:11434/v1 \
  --model llama-3.1-70b \
  --api-key-env MY_GATEWAY_KEY

# Select provider via environment instead of a flag
LLM_PROVIDER=openai onnx-graph-diff a.onnx b.onnx --method llm
```

The existing DashScope path keeps working unchanged, and `--thinking` remains available only on providers that actually support it. New backends can be added programmatically by registering a `ProviderProfile` with `providers.register_provider(...)`.

## Sample report

A `matmul_add_to_gemm` comparison produces a report whose conclusion and region ledger read as:

```
Conclusion: Optimization differences passed static validation
Every detected difference region passed a mathematical-equivalence certificate
under the currently supported ONNX opset rules.

Method: fingerprint   Completed: True
Exact pairs: 1   Difference regions: 1   Unmatched: left 0, right 0

Difference region #1  ── status: proven ── rule: matmul_add_to_gemm
  Left op chain : MatMul -> Add
  Right op chain: Gemm
  Rejoin anchors: left_relu <-> right_relu
  Confidence    : 80%
```

The HTML report additionally renders a model overview (op-type histograms, opsets, inputs/outputs), a per-region evidence panel, an exact-match ledger explaining why each non-region node pair matched, and — on LLM routes — the explanation and per-call token/latency statistics.

## Architecture overview

```
                 ┌─────────────┐      ┌─────────────┐
   left.onnx ───▶│   GraphIR   │      │   GraphIR   │◀─── right.onnx
                 │ (normalized │      │ (normalized │
                 │   IR + FP)  │      │   IR + FP)  │
                 └──────┬──────┘      └──────┬──────┘
                        └───────┬────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │      matching route           │   choose rejoin anchors
                 │  fingerprint | llm | llm-agent│   after each mismatch
                 └──────────────┬────────────────┘
                                ▼  proposed region (per mismatch)
                 ┌──────────────────────────────┐
                 │  StaticEquivalenceValidator   │   independent, opset-aware
                 │   proven / structural /       │   re-derives + checks rule
                 │   rejected / unresolved       │
                 └──────────────┬────────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │   HTML / Markdown / JSON      │
                 └──────────────────────────────┘
```

Key modules under `src/onnx_graph_diff/`:

- **`ir.py`** — loads an ONNX model into a normalized, hashable graph IR with per-node fingerprints and opset validation.
- **`fingerprint.py`** — structural fingerprints used for deterministic exact-matching and anchor enumeration.
- **`matcher.py`** — the deterministic engine: topological readiness, exact-match commitment, region closure, and candidate generation shared by all routes.
- **`context.py`** — serializes bounded windows of the two graphs into the compact JSON the LLM sees.
- **`llm_matcher.py`** / **`agentic_matcher.py`** — the assisted and LLM-led routes; both re-validate every proposal.
- **`equivalence.py`** — the independent opset-rule equivalence checker.
- **`llm_client.py`** + **`providers.py`** — the provider-agnostic LLM client and the provider registry.
- **`workflow.py`** — a LangGraph state machine wiring load → route → match → explain.
- **`report.py`** — the HTML / Markdown / JSON renderers.
- **`cli.py`** — the command-line entry point.

Development-only tooling lives under `scripts/` (fixture generation, a seeded model zoo, rewrite application, and corpus evaluation).

## Development

```bash
pip install -e ".[dev]"

pytest                 # full offline test suite (LLM calls are stubbed with fakes)
ruff check .           # lint
ruff format .          # format
mypy                   # static type check (advisory)
```

A `Makefile` wraps the common targets — `make install`, `make lint`, `make format`, `make test`, `make cov`, and `make check` (the full pre-PR gate). Run `make help` for the full list.

For an end-to-end taste without wiring up your own models, run the bundled example, which generates a fixture pair and diffs it on the deterministic route:

```bash
python examples/run_diff.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution workflow and [CHANGELOG.md](CHANGELOG.md) for release notes.

The test suite runs entirely offline — the LLM routes are exercised through fake clients, so no API key or network access is needed. Continuous integration (GitHub Actions) runs lint, format-check, mypy, and the test suite with coverage across Python 3.10, 3.11, and 3.12.

## Troubleshooting

**Everything comes back `unresolved` / nothing matches.** The tool aligns two graphs by their *structure*, so it assumes `LEFT` and `RIGHT` compute the same thing modulo local rewrites. Passing two unrelated models produces unmatched nodes, not an error. Double-check the argument order (baseline first, optimized second).

**A rewrite I know is valid is reported as `structural` rather than `proven`.** `proven` is only claimed when a supported certificate in `equivalence.py` matches *and* all of its side conditions hold. If the rule exists but a condition fails (e.g. a `Gemm` with `beta != 1`, or a weight that does not byte-match the pre-fusion initializer) the region is `rejected`; if no certificate covers the pattern at all (e.g. a custom-domain fused op) it is `structural`. This is by design — the checker never proves what it cannot certify. See [the equivalence checker](#the-independent-equivalence-checker) for the built-in rules.

**A `proven` rule is unexpectedly `rejected` with an "unavailable in the declared opset" message.** The schema guard refuses to prove a rewrite whose standard-domain operators are not defined in the model's declared opset import. Re-export the model with the correct `opset_import`, or run through `onnx.version_converter` first.

**`RuntimeError: API key is required` on an LLM route.** Only `--method llm`, `--method llm-agent`, and `--explain` need a key. Set the environment variable for your provider (`DASHSCOPE_API_KEY`, `OPENAI_API_KEY`, or a custom one via `--api-key-env`) — see [LLM providers](#llm-providers). The default `--method fingerprint` route needs no key.

**`No model configured` for `--provider openai-compatible`.** The generic backend has no default model; pass `--model` (and usually `--base-url`).

**Shapes look wrong or missing in the report.** Shape inference is best-effort and silently skipped if it fails, so some edges may show no dtype/shape. Matching still works from structure; the shapes are informational.

Not covered here? Open an issue using the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) with the two models, the exact command, and the printed JSON summary.

## License

[MIT](LICENSE) © 2026 Azimbek Olimbekov
