from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx

from onnx_graph_diff.rewrite import RULES, RuleEngine


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply static ONNX rewrite rules to create test variants.")
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--rules", help="Comma-separated rule IDs in application order")
    parser.add_argument("--manifest", help="Optional JSON lineage output path")
    parser.add_argument("--once", action="store_true", help="Apply each rule at most once")
    parser.add_argument("--list-rules", action="store_true")
    args = parser.parse_args()
    if args.list_rules:
        print(json.dumps({"rules": sorted(RULES)}, indent=2))
        return 0
    if not args.input or not args.output or not args.rules:
        parser.error("input, output, and --rules are required unless --list-rules is used")

    rule_ids = _csv(args.rules)
    unknown = sorted(set(rule_ids) - set(RULES))
    if unknown:
        parser.error("unknown rule IDs: " + ", ".join(unknown))
    outcome = RuleEngine().apply(
        onnx.load(args.input),
        rule_ids,
        repeat_each_rule=not args.once,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(outcome.model, output)
    payload = {
        "input": args.input,
        "output": str(output),
        "requested_rules": rule_ids,
        "applications": [record.to_dict() for record in outcome.records],
    }
    if args.manifest:
        manifest = Path(args.manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
