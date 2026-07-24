"""Command-line entry point for ``onnx-graph-diff``.

Parses the two model paths and the matching options, assembles the matcher and
context configs, builds the LangGraph workflow, and runs load -> route -> match
-> (optional) explain. It then renders the requested report format (with a JSON
sidecar by default) and prints a compact summary -- conclusion, region count,
and unmatched counts -- to stdout as JSON.

:func:`main` returns a process exit code and is wired to the ``onnx-graph-diff``
console script in ``pyproject.toml``. An :class:`~argparse.ArgumentParser`
suitable for docs or testing is available via :func:`build_parser`.
"""

from __future__ import annotations

import argparse
import json

from .agentic_matcher import AgenticMatcherConfig
from .context import ContextConfig
from .llm_client import LLMClient, LLMConfig
from .llm_matcher import LLMMatcherConfig
from .matcher import MatcherConfig
from .providers import PROVIDER_CHOICES, provider_from_env
from .report import build_report_data, default_report_path, write_comparison_report
from .workflow import build_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Statically compare two existing ONNX graphs and generate a readable report.",
        epilog="Model generation, rewrite, and corpus evaluation are development scripts under scripts/.",
    )
    parser.add_argument("left", help="Path to the baseline or unoptimized ONNX model")
    parser.add_argument("right", help="Path to the candidate or optimized ONNX model")
    parser.add_argument("--method", choices=["fingerprint", "llm", "llm-agent"], default="fingerprint")
    parser.add_argument(
        "--report",
        help="Report output path. Defaults to reports/<left>__vs__<right>.html.",
    )
    parser.add_argument(
        "--report-format",
        choices=["html", "markdown", "json", "all"],
        default="html",
        help="Primary report format; HTML is standalone and is the default.",
    )
    parser.add_argument(
        "--json-sidecar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the complete machine-readable report beside HTML/Markdown.",
    )
    parser.add_argument(
        "--explain",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ask the LLM for a post-match natural-language explanation. Defaults on for LLM methods.",
    )
    parser.add_argument(
        "--reject-structural-unknown",
        action="store_true",
        help="Leave unknown transformations unresolved instead of accepting structural-only rejoins.",
    )
    _add_llm_args(parser)
    return parser


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("LLM matching and explanation")
    group.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=None,
        help=(
            "LLM backend for --method llm/llm-agent and --explain. "
            "Defaults to the LLM_PROVIDER env var, else 'dashscope'. "
            "'dashscope'/'qwen' use DASHSCOPE_API_KEY, 'openai' uses OPENAI_API_KEY, "
            "'openai-compatible' targets any OpenAI-style endpoint via --base-url/--model/--api-key-env."
        ),
    )
    group.add_argument(
        "--model",
        default=None,
        help="Chat model name. Defaults to the selected provider's default model.",
    )
    group.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL. Defaults to the selected provider's endpoint.",
    )
    group.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable to read the API key from (overrides the provider default).",
    )
    group.add_argument("--thinking", action="store_true")
    group.add_argument("--temperature", type=float, default=0.0)
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--timeout", type=float, default=120.0)
    group.add_argument(
        "--context-strategy",
        choices=["parents_only", "bounded", "path_summary", "full_window"],
        default="path_summary",
    )
    group.add_argument(
        "--prompt-variant",
        choices=["anchor_v1", "proof_v2", "region_v3", "region_v4"],
        default="region_v4",
    )
    group.add_argument("--max-hops", type=int, default=8)
    group.add_argument("--max-nodes", type=int, default=20)
    group.add_argument("--max-chars", type=int, default=24000)
    group.add_argument("--max-retries", type=int, default=1)
    group.add_argument("--no-coarse-candidates", action="store_true")
    group.add_argument(
        "--agent-prompt",
        choices=[
            "agent_v1",
            "agent_v2",
            "agent_v3",
            "agent_v4",
            "agent_v5",
            "agent_v6",
            "agent_v7",
            "agent_v8",
            "agent_v9",
            "agent_v10",
            "agent_v11",
            "agent_v12",
            "agent_v13",
        ],
        default="agent_v13",
        help="Prompt used by --method llm-agent.",
    )
    group.add_argument(
        "--agent-context-strategy",
        choices=["bounded", "path_summary", "full_window"],
        default="bounded",
    )
    group.add_argument("--agent-initial-hops", type=int, default=2)
    group.add_argument("--agent-initial-nodes", type=int, default=8)
    group.add_argument("--agent-max-expansions", type=int, default=3)
    group.add_argument("--agent-max-calls", type=int, default=8)


def _llm_config(args: argparse.Namespace) -> LLMConfig:
    return LLMConfig(
        provider=provider_from_env(args.provider),
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        thinking=args.thinking,
        temperature=args.temperature,
        seed=args.seed,
        timeout_seconds=args.timeout,
    )


def _context_config(args: argparse.Namespace) -> ContextConfig:
    return ContextConfig(
        strategy=args.context_strategy,
        max_hops=args.max_hops,
        max_nodes_per_graph=args.max_nodes,
        max_chars=args.max_chars,
        include_coarse_anchor_pairs=not args.no_coarse_candidates,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    method = args.method.replace("-", "_")
    explain = args.explain if args.explain is not None else method in {"llm", "llm_agent"}
    client = LLMClient(_llm_config(args)) if method in {"llm", "llm_agent"} or explain else None
    allow_structural_unknown = not args.reject_structural_unknown
    workflow = build_workflow(
        llm_client=client,
        fingerprint_config=MatcherConfig(allow_structural_unknown=allow_structural_unknown),
        llm_config=LLMMatcherConfig(
            context=_context_config(args),
            prompt_variant=args.prompt_variant,
            max_retries=args.max_retries,
            allow_structural_unknown=allow_structural_unknown,
            explain=False,
        ),
        agentic_config=AgenticMatcherConfig(
            context_strategy=args.agent_context_strategy,
            prompt_variant=args.agent_prompt,
            initial_hops=args.agent_initial_hops,
            initial_nodes_per_graph=args.agent_initial_nodes,
            max_hops=args.max_hops,
            max_nodes_per_graph=args.max_nodes,
            max_chars=args.max_chars,
            max_expansions=args.agent_max_expansions,
            max_retries_per_mismatch=args.max_retries,
            max_calls_per_mismatch=args.agent_max_calls,
            allow_structural_unknown=allow_structural_unknown,
            explain=False,
        ),
    )
    state = workflow.invoke(
        {
            "left_path": args.left,
            "right_path": args.right,
            "method": method,
            "explain": explain,
        }
    )
    result = state["result"]
    result.metadata.update(
        {
            "left_path": args.left,
            "right_path": args.right,
            "explain": explain,
            "provider": client.profile.name if client is not None else None,
            "model": client.model if client is not None else None,
            "thinking": args.thinking if client is not None else None,
            "context_strategy": (
                args.agent_context_strategy
                if method == "llm_agent"
                else args.context_strategy
                if method == "llm"
                else None
            ),
            "prompt_variant": args.agent_prompt
            if method == "llm_agent"
            else args.prompt_variant
            if method == "llm"
            else None,
            "max_hops": args.max_hops,
            "max_nodes": args.max_nodes,
            "max_chars": args.max_chars,
            "max_retries": args.max_retries if method in {"llm", "llm_agent"} else None,
            "agent_initial_hops": args.agent_initial_hops if method == "llm_agent" else None,
            "agent_initial_nodes": args.agent_initial_nodes if method == "llm_agent" else None,
            "agent_max_expansions": args.agent_max_expansions if method == "llm_agent" else None,
            "agent_max_calls": args.agent_max_calls if method == "llm_agent" else None,
        }
    )
    report_data = build_report_data(
        state["left_graph"],
        state["right_graph"],
        result,
        left_path=args.left,
        right_path=args.right,
    )
    output = args.report or default_report_path(args.left, args.right, fmt=args.report_format)
    written = write_comparison_report(
        report_data,
        output,
        report_format=args.report_format,
        json_sidecar=args.json_sidecar,
    )
    summary = {
        "conclusion": report_data["conclusion"],
        "completed": result.completed,
        "regions": report_data["comparison"]["region_count"],
        "unmatched_left": len(result.unmatched_left),
        "unmatched_right": len(result.unmatched_right),
        "reports": {name: str(path) for name, path in written.items()},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
