from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .ir import GraphIR
from .models import ComparisonResult


def load_corpus(corpus_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(corpus_dir)
    return json.loads((path / "corpus.json").read_text(encoding="utf-8"))


def score_result(manifest: dict[str, Any], result: ComparisonResult) -> dict[str, Any]:
    regions = [event for event in result.events if event.kind == "region"]
    region = regions[0] if regions else None
    expected_regions = manifest.get(
        "regions",
        [
            {
                "left_anchor": manifest["left_anchor"],
                "right_anchor": manifest["right_anchor"],
                "left_region": manifest["left_region"],
                "right_region": manifest["right_region"],
                "rule_id": manifest["rule_id"],
                "proven": manifest.get("proven", False),
            }
        ],
    )
    expected_equivalent = bool(manifest.get("equivalent", True))
    false_positive = not expected_equivalent and region is not None
    if expected_equivalent:
        anchor_correct = len(regions) == len(expected_regions) and all(
            actual.left_anchor == expected["left_anchor"] and actual.right_anchor == expected["right_anchor"]
            for actual, expected in zip(regions, expected_regions, strict=False)
        )
        region_correct = len(regions) == len(expected_regions) and all(
            actual.left_nodes == expected["left_region"] and actual.right_nodes == expected["right_region"]
            for actual, expected in zip(regions, expected_regions, strict=False)
        )
    else:
        anchor_correct = not false_positive
        region_correct = not false_positive
    validated_rule = region.proof.rule_id if region and region.proof else None
    claimed_rules: list[str | None] = []
    match_calls = [call for call in result.llm_calls if call.context_strategy != "explanation"]
    accepted_calls = [call for call in match_calls if call.metadata.get("accepted") is True]
    expansion_calls = [call for call in match_calls if call.metadata.get("action") == "expand"]
    for call in accepted_calls:
        claimed_rules.append(call.metadata.get("claimed_rule"))
    claimed_rule = claimed_rules[0] if claimed_rules else None
    if not expected_equivalent:
        strict_claimed_rule_correct = not accepted_calls
    else:
        strict_claimed_rule_correct = len(claimed_rules) == len(expected_regions) and all(
            claimed == expected["rule_id"] for claimed, expected in zip(claimed_rules, expected_regions, strict=False)
        )
    selected_anchor_was_self_scanned = True
    if result.method == "llm_agent" and expected_equivalent:
        selected_anchor_was_self_scanned = len(accepted_calls) == len(expected_regions) and all(
            (
                str(call.metadata.get("anchor_pair", [""])[0]).startswith("__graph_output__:")
                or call.metadata.get("anchor_pair") in call.metadata.get("same_op_pairs_seen", [])
            )
            for call in accepted_calls
        )
    if not expected_equivalent:
        validated_rule_correct = not false_positive
        claimed_rule_correct = not false_positive
    elif len(regions) == len(expected_regions):
        validated_rule_correct = all(
            (actual.proof.rule_id if actual.proof else None) == expected["rule_id"] if expected.get("proven") else True
            for actual, expected in zip(regions, expected_regions, strict=False)
        )
        claimed_rule_correct = len(claimed_rules) == len(expected_regions) and all(
            claimed == expected["rule_id"] for claimed, expected in zip(claimed_rules, expected_regions, strict=False)
        )
    else:
        validated_rule_correct = False
        claimed_rule_correct = False
    is_llm_method = result.method in {"llm", "llm_agent"}
    if not is_llm_method:
        claimed_rule_correct = validated_rule_correct
        proposed_region_correct = region_correct
    elif not expected_equivalent:
        proposed_region_correct = not false_positive
    else:
        proposed_region_correct = len(accepted_calls) == len(expected_regions) and all(
            call.metadata.get("proposed_left_region") == expected["left_region"]
            and call.metadata.get("proposed_right_region") == expected["right_region"]
            for call, expected in zip(accepted_calls, expected_regions, strict=False)
        )
    parse_ok = any(call.parse_ok for call in match_calls) if match_calls else True
    first_parse_ok = match_calls[0].parse_ok if match_calls else True
    call_parse_rate = sum(call.parse_ok for call in match_calls) / len(match_calls) if match_calls else 1.0
    total_tokens = sum(call.total_tokens or 0 for call in match_calls)
    latency = sum(call.latency_seconds or 0.0 for call in match_calls)
    return {
        "case": manifest["name"],
        "method": result.method,
        "completed": result.completed,
        "expected_equivalent": expected_equivalent,
        "anchor_correct": anchor_correct,
        "region_correct": region_correct,
        "validated_rule": validated_rule,
        "claimed_rule": claimed_rule,
        "rule_correct": claimed_rule_correct if is_llm_method else validated_rule_correct,
        "claimed_rule_correct": claimed_rule_correct,
        "strict_claimed_rule_correct": strict_claimed_rule_correct,
        "selected_anchor_was_self_scanned": selected_anchor_was_self_scanned,
        "validated_rule_correct": validated_rule_correct,
        "llm_proposed_region_correct": proposed_region_correct,
        "proof_status": region.proof.status if region and region.proof else None,
        "all_expected_regions_proven": any(expected.get("proven") for expected in expected_regions)
        and all(
            event.proof is not None and event.proof.status == "proven"
            for event, expected in zip(regions, expected_regions, strict=False)
            if expected.get("proven")
        ),
        "false_positive": false_positive,
        "outcome_correct": (
            result.completed and anchor_correct and region_correct if expected_equivalent else not false_positive
        ),
        "llm_parse_ok": parse_ok,
        "llm_first_parse_ok": first_parse_ok,
        "llm_call_parse_rate": call_parse_rate,
        "retry_count": max(0, len(match_calls) - len(accepted_calls) - len(expansion_calls)),
        "expansion_count": len(expansion_calls),
        "llm_calls": len(match_calls),
        "prompt_tokens": sum(call.prompt_tokens or 0 for call in match_calls),
        "completion_tokens": sum(call.completion_tokens or 0 for call in match_calls),
        "total_tokens": total_tokens,
        "latency_seconds": latency,
        "metadata": result.metadata,
        "call_details": [call.metadata for call in match_calls],
    }


def evaluate_corpus(
    corpus_dir: str | Path,
    matcher_factory: Callable[[dict[str, Any]], Any],
    *,
    case_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    corpus_path = Path(corpus_dir)
    records: list[dict[str, Any]] = []
    for manifest in load_corpus(corpus_path):
        if case_names is not None and manifest["name"] not in case_names:
            continue
        matcher = matcher_factory(manifest)
        result = matcher.compare(
            GraphIR.from_path(str(corpus_path / manifest["left"])),
            GraphIR.from_path(str(corpus_path / manifest["right"])),
        )
        records.append(score_result(manifest, result))
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"cases": 0}

    def rate(field: str) -> float:
        return sum(bool(record[field]) for record in records) / len(records)

    return {
        "cases": len(records),
        "completion_rate": rate("completed"),
        "outcome_accuracy": rate("outcome_correct"),
        "anchor_accuracy": rate("anchor_correct"),
        "region_accuracy": rate("region_correct"),
        "rule_accuracy": rate("rule_correct"),
        "claimed_rule_accuracy": rate("claimed_rule_correct"),
        "strict_claimed_rule_accuracy": rate("strict_claimed_rule_correct"),
        "self_scanned_anchor_rate": rate("selected_anchor_was_self_scanned"),
        "validated_rule_accuracy": rate("validated_rule_correct"),
        "llm_proposed_region_accuracy": rate("llm_proposed_region_correct"),
        "parse_rate": rate("llm_parse_ok"),
        "first_parse_rate": rate("llm_first_parse_ok"),
        "call_parse_rate": statistics.mean(record["llm_call_parse_rate"] for record in records),
        "false_positive_rate": rate("false_positive"),
        "proven_rate": rate("all_expected_regions_proven"),
        "avg_total_tokens": statistics.mean(record["total_tokens"] for record in records),
        "avg_latency_seconds": statistics.mean(record["latency_seconds"] for record in records),
        "avg_expansions": statistics.mean(record.get("expansion_count", 0) for record in records),
        "avg_retries": statistics.mean(record.get("retry_count", 0) for record in records),
    }


def summarize_by_experiment(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    descriptions: dict[str, dict[str, Any]] = {}
    for record in records:
        experiment = record.get("experiment", {})
        key = json.dumps(experiment, sort_keys=True, separators=(",", ":"))
        groups[key].append(record)
        descriptions[key] = experiment
    return [
        {"experiment": descriptions[key], "summary": summarize(group_records)}
        for key, group_records in sorted(groups.items())
    ]


def write_report(path: str | Path, records: list[dict[str, Any]], *, experiment: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "summary": summarize(records),
        "groups": summarize_by_experiment(records),
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
