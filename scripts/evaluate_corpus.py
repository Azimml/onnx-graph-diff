from __future__ import annotations

import argparse
import json

from onnx_graph_diff.agentic_matcher import AgenticMatcherConfig, LLMAgenticMatcher
from onnx_graph_diff.context import ContextConfig
from onnx_graph_diff.evaluation import evaluate_corpus, summarize, write_report
from onnx_graph_diff.llm_client import LLMClient, LLMConfig
from onnx_graph_diff.llm_matcher import LLMHeuristicMatcher, LLMMatcherConfig
from onnx_graph_diff.matcher import FingerprintMatcher, MatcherConfig
from onnx_graph_diff.providers import PROVIDER_CHOICES, provider_from_env


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool_csv(value: str) -> list[bool]:
    result = []
    for item in _csv(value):
        if item.lower() in {"true", "1", "yes"}:
            result.append(True)
        elif item.lower() in {"false", "0", "no"}:
            result.append(False)
        else:
            raise argparse.ArgumentTypeError(f"Invalid boolean value: {item}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate graph matchers on a generated test corpus.")
    parser.add_argument("--corpus", default="artifacts/fixtures")
    parser.add_argument("--method", choices=["fingerprint", "llm", "llm-agent"], default="fingerprint")
    parser.add_argument("--output", default="reports/evaluation.json")
    parser.add_argument("--cases", default="")
    parser.add_argument("--strategies", default="path_summary")
    parser.add_argument("--prompts", default="region_v4")
    parser.add_argument("--agent-strategies", default="bounded")
    parser.add_argument("--agent-prompts", default="agent_v13")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--thinking-modes", default="false")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--capture-transcripts", action="store_true")
    parser.add_argument("--provider", choices=PROVIDER_CHOICES, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--agent-initial-hops", type=int, default=2)
    parser.add_argument("--agent-initial-nodes", type=int, default=8)
    parser.add_argument("--agent-max-expansions", type=int, default=3)
    parser.add_argument("--agent-max-calls", type=int, default=8)
    parser.add_argument("--no-coarse-candidates", action="store_true")
    parser.add_argument("--reject-structural-unknown", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = set(_csv(args.cases)) or None
    allow_structural_unknown = not args.reject_structural_unknown
    if args.method == "fingerprint":
        records = evaluate_corpus(
            args.corpus,
            lambda _: FingerprintMatcher(MatcherConfig(allow_structural_unknown=allow_structural_unknown)),
            case_names=cases,
        )
        write_report(args.output, records, experiment={"method": "fingerprint"})
        print(json.dumps(summarize(records), indent=2))
        return 0

    all_records = []
    is_agentic = args.method == "llm-agent"
    strategies = _csv(args.agent_strategies if is_agentic else args.strategies)
    prompts = _csv(args.agent_prompts if is_agentic else args.prompts)
    seeds = [int(value) for value in _csv(args.seeds)]
    thinking_modes = _bool_csv(args.thinking_modes)
    for repeat in range(args.repeats):
        for thinking in thinking_modes:
            for strategy in strategies:
                for prompt in prompts:
                    for seed in seeds:
                        client = LLMClient(
                            LLMConfig(
                                provider=provider_from_env(args.provider),
                                model=args.model,
                                base_url=args.base_url,
                                api_key_env=args.api_key_env,
                                thinking=thinking,
                                temperature=args.temperature,
                                seed=seed,
                                timeout_seconds=args.timeout,
                            )
                        )
                        if is_agentic:
                            config = AgenticMatcherConfig(
                                context_strategy=strategy,
                                prompt_variant=prompt,
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
                                capture_transcripts=args.capture_transcripts,
                            )

                            def matcher_factory(_, client=client, config=config):
                                return LLMAgenticMatcher(client, config)
                        else:
                            config = LLMMatcherConfig(
                                context=ContextConfig(
                                    strategy=strategy,
                                    max_hops=args.max_hops,
                                    max_nodes_per_graph=args.max_nodes,
                                    max_chars=args.max_chars,
                                    include_coarse_anchor_pairs=not args.no_coarse_candidates,
                                ),
                                prompt_variant=prompt,
                                max_retries=args.max_retries,
                                allow_structural_unknown=allow_structural_unknown,
                                explain=False,
                                capture_transcripts=args.capture_transcripts,
                            )

                            def matcher_factory(_, client=client, config=config):
                                return LLMHeuristicMatcher(client, config)

                        records = evaluate_corpus(
                            args.corpus,
                            matcher_factory,
                            case_names=cases,
                        )
                        for record in records:
                            record["experiment"] = {
                                "repeat": repeat,
                                "thinking": thinking,
                                "strategy": strategy,
                                "prompt": prompt,
                                "seed": seed,
                            }
                        all_records.extend(records)
    experiment = {
        "method": "llm_agent" if is_agentic else "llm",
        "model": args.model,
        "strategies": strategies,
        "prompts": prompts,
        "seeds": seeds,
        "thinking_modes": thinking_modes,
        "repeats": args.repeats,
    }
    if is_agentic:
        experiment.update(
            {
                "initial_hops": args.agent_initial_hops,
                "initial_nodes": args.agent_initial_nodes,
                "max_expansions": args.agent_max_expansions,
                "max_calls": args.agent_max_calls,
                "coarse_candidates_supplied": False,
            }
        )
    write_report(args.output, all_records, experiment=experiment)
    print(json.dumps(summarize(all_records), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
