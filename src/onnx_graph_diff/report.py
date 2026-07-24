from __future__ import annotations

import html
import json
import re
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ir import GraphIR
from .models import ComparisonResult, MatchEvent

REPORT_SCHEMA_VERSION = 2


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _graph_summary(graph: GraphIR, path: str) -> dict[str, Any]:
    op_counts = Counter(graph.nodes[node_id].op_type for node_id in graph.node_order)
    return {
        "name": graph.name or Path(path).stem,
        "path": str(Path(path).resolve()),
        "node_count": len(graph.nodes),
        "value_count": len(graph.values),
        "initializer_count": sum(value.initializer is not None for value in graph.values.values()),
        "inputs": [graph.values[name].compact() for name in graph.graph_inputs],
        "outputs": [graph.values[name].compact() for name in graph.graph_outputs],
        "opsets": graph.opsets,
        "op_histogram": dict(sorted(op_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def _node_details(graph: GraphIR, node_ids: Iterable[str]) -> list[dict[str, Any]]:
    return [graph.compact_node(node_id) for node_id in node_ids if node_id in graph.nodes]


def _proof_status(event: MatchEvent) -> str:
    if event.kind == "unresolved":
        return "unresolved"
    if event.proof is None:
        return "unverified"
    return event.proof.status


def _conclusion(result: ComparisonResult) -> dict[str, str]:
    region_events = [event for event in result.events if event.kind == "region"]
    statuses = {_proof_status(event) for event in region_events}
    has_unresolved = any(event.kind == "unresolved" for event in result.events)
    if not result.completed or has_unresolved or result.unmatched_left or result.unmatched_right:
        return {
            "code": "unresolved",
            "label": "Unresolved differences remain",
            "detail": (
                "Graph traversal did not fully rematch. The unmatched nodes in this report need a new "
                "rule, a wider static-proof capability, or manual review."
            ),
        }
    if "rejected" in statuses:
        return {
            "code": "rejected",
            "label": "Candidate optimization failed static validation",
            "detail": (
                "A structural candidate violated the conditions of a known rule, so the two models cannot "
                "be declared mathematically equivalent in that region."
            ),
        }
    if statuses & {"structural", "hypothesis", "unverified"}:
        return {
            "code": "structural",
            "label": "Structurally rematched, equivalence not fully proven",
            "detail": (
                "Every node found a downstream rejoin point, but at least one region lacks a supported "
                "ONNX mathematical-equivalence certificate."
            ),
        }
    if region_events:
        return {
            "code": "proven",
            "label": "Optimization differences passed static validation",
            "detail": (
                "Every detected difference region passed a mathematical-equivalence certificate under the "
                "currently supported ONNX opset rules."
            ),
        }
    return {
        "code": "identical",
        "label": "No structural differences found",
        "detail": "The two graphs matched node-for-node under the current normalization and fingerprint rules.",
    }


def build_report_data(
    left: GraphIR,
    right: GraphIR,
    result: ComparisonResult,
    *,
    left_path: str,
    right_path: str,
) -> dict[str, Any]:
    exact_events = [event for event in result.events if event.kind == "exact"]
    region_events = [event for event in result.events if event.kind == "region"]
    unresolved_events = [event for event in result.events if event.kind == "unresolved"]
    exact_pairs = [
        {
            "index": index,
            "source": event.source,
            "left_node_id": event.left_nodes[0] if event.left_nodes else None,
            "right_node_id": event.right_nodes[0] if event.right_nodes else None,
            "left_node": _node_details(left, event.left_nodes),
            "right_node": _node_details(right, event.right_nodes),
            "output_mapping": [list(pair) for pair in event.value_mapping],
            "match_reason": event.explanation,
        }
        for index, event in enumerate(exact_events, start=1)
    ]
    regions: list[dict[str, Any]] = []
    for index, event in enumerate(region_events, start=1):
        left_nodes = _node_details(left, event.left_nodes)
        right_nodes = _node_details(right, event.right_nodes)
        regions.append(
            {
                "index": index,
                "status": _proof_status(event),
                "rule_id": event.proof.rule_id if event.proof else None,
                "source": event.source,
                "confidence": event.confidence,
                "left_anchor": event.left_anchor,
                "right_anchor": event.right_anchor,
                "left_node_ids": event.left_nodes,
                "right_node_ids": event.right_nodes,
                "left_op_chain": [node["op"] for node in left_nodes],
                "right_op_chain": [node["op"] for node in right_nodes],
                "left_nodes": left_nodes,
                "right_nodes": right_nodes,
                "boundary_mapping": [list(pair) for pair in event.value_mapping],
                "proof_details": event.proof.details if event.proof else [],
                "heuristic_explanation": event.explanation,
            }
        )

    llm_calls = [_jsonable(call.__dict__) for call in result.llm_calls]
    total_tokens = sum(call.total_tokens or 0 for call in result.llm_calls)
    total_latency = sum(call.latency_seconds or 0.0 for call in result.llm_calls)
    parse_failures = sum(not call.parse_ok for call in result.llm_calls)
    status_counts = Counter(region["status"] for region in regions)

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conclusion": _conclusion(result),
        "comparison": {
            "method": result.method,
            "completed": result.completed,
            "configuration": _jsonable(result.metadata),
            "exact_pair_count": len(exact_events),
            "exact_left_node_count": sum(len(event.left_nodes) for event in exact_events),
            "exact_right_node_count": sum(len(event.right_nodes) for event in exact_events),
            "region_count": len(region_events),
            "region_status_counts": dict(status_counts),
            "unresolved_event_count": len(unresolved_events),
            "unmatched_left_count": len(result.unmatched_left),
            "unmatched_right_count": len(result.unmatched_right),
        },
        "left_model": _graph_summary(left, left_path),
        "right_model": _graph_summary(right, right_path),
        "regions": regions,
        "exact_pairs": exact_pairs,
        "unmatched": {
            "left": _node_details(left, result.unmatched_left),
            "right": _node_details(right, result.unmatched_right),
            "frontiers": [
                {
                    "left": _node_details(left, event.left_nodes),
                    "right": _node_details(right, event.right_nodes),
                }
                for event in unresolved_events
            ],
        },
        "value_mapping": result.value_mapping,
        "llm": {
            "call_count": len(result.llm_calls),
            "total_tokens": total_tokens,
            "total_latency_seconds": total_latency,
            "parse_failures": parse_failures,
            "calls": llm_calls,
            "final_explanation": result.explanation,
        },
        "raw_result": _jsonable(result.to_dict()),
    }
    return _jsonable(payload)


def _fmt_chain(ops: list[str]) -> str:
    return " -> ".join(ops) if ops else "(empty / eliminated)"


def _fmt_percent(value: float) -> str:
    return f"{max(0.0, min(1.0, value)) * 100:.0f}%"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_None_"

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = ["| " + " | ".join(map(clean, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(clean, row)) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    conclusion = data["conclusion"]
    comparison = data["comparison"]
    left = data["left_model"]
    right = data["right_model"]
    lines = [
        "# ONNX Graph Difference Report",
        "",
        f"**Conclusion: {conclusion['label']}**",
        "",
        conclusion["detail"],
        "",
        "## Overview",
        "",
        _markdown_table(
            ["Item", "Left", "Right"],
            [
                ["Model", left["name"], right["name"]],
                ["Path", left["path"], right["path"]],
                ["Nodes", str(left["node_count"]), str(right["node_count"])],
                [
                    "Inputs / outputs",
                    f"{len(left['inputs'])} / {len(left['outputs'])}",
                    f"{len(right['inputs'])} / {len(right['outputs'])}",
                ],
                [
                    "Opsets",
                    json.dumps(left["opsets"], ensure_ascii=False),
                    json.dumps(right["opsets"], ensure_ascii=False),
                ],
            ],
        ),
        "",
        f"Method: `{comparison['method']}`  ",
        f"Completed: `{comparison['completed']}`  ",
        f"Exact pairs: `{comparison['exact_pair_count']}`  ",
        f"Difference regions: `{comparison['region_count']}`  ",
        f"Unmatched nodes: left `{comparison['unmatched_left_count']}`, right `{comparison['unmatched_right_count']}`",
        "",
        "## Exact Match Ledger",
        "",
        _markdown_table(
            ["#", "Left node", "Right node", "Why it matched"],
            [
                [
                    str(pair["index"]),
                    pair["left_node_id"] or "-",
                    pair["right_node_id"] or "-",
                    pair["match_reason"] or "Exact structural and mapped-input checks passed.",
                ]
                for pair in data["exact_pairs"]
            ],
        ),
        "",
        "## Difference Regions",
        "",
    ]
    region_rows = []
    for region in data["regions"]:
        region_rows.append(
            [
                str(region["index"]),
                region["status"],
                region["rule_id"] or "unknown",
                _fmt_chain(region["left_op_chain"]),
                _fmt_chain(region["right_op_chain"]),
                f"{region['left_anchor']} <-> {region['right_anchor']}",
                _fmt_percent(region["confidence"]),
            ]
        )
    lines.extend(
        [
            _markdown_table(
                ["#", "Status", "Rule", "Left", "Right", "Rejoin", "Confidence"],
                region_rows,
            ),
            "",
        ]
    )
    for region in data["regions"]:
        lines.extend(
            [
                f"### Region {region['index']}: {region['rule_id'] or 'unknown'}",
                "",
                f"- Status: `{region['status']}`",
                f"- Source: `{region['source']}`",
                f"- Left nodes: `{', '.join(region['left_node_ids']) or '(empty)'}`",
                f"- Right nodes: `{', '.join(region['right_node_ids']) or '(empty)'}`",
                f"- Boundary mapping: `{json.dumps(region['boundary_mapping'], ensure_ascii=False)}`",
                f"- Proof evidence: {'; '.join(region['proof_details']) or 'No static certificate.'}",
            ]
        )
        if region["heuristic_explanation"]:
            lines.append(f"- LLM heuristic: {region['heuristic_explanation']}")
        lines.extend(
            [
                "",
                "```json",
                json.dumps(
                    {"left": region["left_nodes"], "right": region["right_nodes"]}, indent=2, ensure_ascii=False
                ),
                "```",
                "",
            ]
        )

    lines.extend(["## LLM Summary", ""])
    llm = data["llm"]
    lines.append(
        f"Calls: `{llm['call_count']}`, tokens: `{llm['total_tokens']}`, latency: `{llm['total_latency_seconds']:.2f}s`, parse failures: `{llm['parse_failures']}`."
    )
    if llm["final_explanation"]:
        lines.extend(["", llm["final_explanation"]])
    if comparison["unmatched_left_count"] or comparison["unmatched_right_count"]:
        lines.extend(
            [
                "",
                "## Unmatched Nodes",
                "",
                "```json",
                json.dumps(data["unmatched"], indent=2, ensure_ascii=False),
                "```",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json_pre(value: Any) -> str:
    return _e(json.dumps(value, indent=2, ensure_ascii=False))


def _node_table(nodes: list[dict[str, Any]]) -> str:
    if not nodes:
        return '<p class="muted">Empty side (operation eliminated or inserted).</p>'
    rows = []
    for node in nodes:
        inputs = [_value_label(item) for item in node.get("inputs", [])]
        outputs = [_value_label(item) for item in node.get("outputs", [])]
        rows.append(
            "<tr>"
            f"<td><code>{_e(node['id'])}</code></td>"
            f"<td><code>{_e(node['op'])}</code></td>"
            f"<td>{_e(', '.join(inputs) or '-')}</td>"
            f"<td>{_e(', '.join(outputs) or '-')}</td>"
            f"<td><pre>{_json_pre(node.get('attrs', {}))}</pre></td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Node</th><th>Op</th><th>Inputs</th><th>Outputs</th><th>Attributes</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _value_label(value: dict[str, Any]) -> str:
    parts = [value["name"]]
    detail = []
    if value.get("dtype"):
        detail.append(str(value["dtype"]))
    if "shape" in value:
        detail.append(str(value["shape"]))
    if detail:
        parts.append("[" + ", ".join(detail) + "]")
    const = value.get("const")
    if const:
        const_text = f"const {const.get('dtype')} {const.get('shape')} #{const.get('hash')}"
        if "scalar" in const:
            const_text += f" = {const['scalar']}"
        parts.append(const_text)
    return " ".join(parts)


def _value_table(values: list[dict[str, Any]]) -> str:
    if not values:
        return '<p class="muted">None</p>'
    rows = "".join(
        f"<tr><td><code>{_e(value['name'])}</code></td><td>{_e(value.get('dtype', '-'))}</td>"
        f"<td><code>{_e(value.get('shape', '-'))}</code></td></tr>"
        for value in values
    )
    return f'<div class="table-wrap"><table><thead><tr><th>Value</th><th>Dtype</th><th>Shape</th></tr></thead><tbody>{rows}</tbody></table></div>'


def _histogram_table(histogram: dict[str, int]) -> str:
    return "".join(f"<tr><td><code>{_e(op)}</code></td><td>{count}</td></tr>" for op, count in histogram.items())


def render_html(data: dict[str, Any]) -> str:
    conclusion = data["conclusion"]
    comparison = data["comparison"]
    left = data["left_model"]
    right = data["right_model"]
    region_rows = []
    region_details = []
    exact_rows = []
    for pair in data["exact_pairs"]:
        left_node = pair["left_node"][0] if pair["left_node"] else {}
        right_node = pair["right_node"][0] if pair["right_node"] else {}
        exact_rows.append(
            "<tr>"
            f"<td>{pair['index']}</td>"
            f'<td><code>{_e(pair["left_node_id"])}</code><br><span class="muted">{_e(left_node.get("op", "-"))}</span></td>'
            f'<td><code>{_e(pair["right_node_id"])}</code><br><span class="muted">{_e(right_node.get("op", "-"))}</span></td>'
            f"<td>{_e(pair['match_reason'] or 'Exact structural and mapped-input checks passed.')}</td>"
            f"<td><pre>{_json_pre(pair['output_mapping'])}</pre></td>"
            "</tr>"
        )
    for region in data["regions"]:
        status = region["status"]
        region_rows.append(
            "<tr>"
            f"<td>{region['index']}</td>"
            f'<td><span class="badge {_e(status)}">{_e(status)}</span></td>'
            f"<td><code>{_e(region['rule_id'] or 'unknown')}</code></td>"
            f"<td>{_e(_fmt_chain(region['left_op_chain']))}</td>"
            f"<td>{_e(_fmt_chain(region['right_op_chain']))}</td>"
            f"<td><code>{_e(region['left_anchor'])}</code><br><code>{_e(region['right_anchor'])}</code></td>"
            f"<td>{_e(_fmt_percent(region['confidence']))}</td>"
            "</tr>"
        )
        evidence = "".join(f"<li>{_e(item)}</li>" for item in region["proof_details"])
        if not evidence:
            evidence = '<li class="muted">No static certificate was produced.</li>'
        heuristic = ""
        if region["heuristic_explanation"]:
            heuristic = (
                f'<div class="note"><strong>LLM heuristic</strong><p>{_e(region["heuristic_explanation"])}</p></div>'
            )
        region_details.append(
            f'<details class="region" open><summary><span>Region {region["index"]}</span>'
            f"<code>{_e(region['rule_id'] or 'unknown')}</code>"
            f'<span class="badge {_e(status)}">{_e(status)}</span></summary>'
            '<div class="region-body">'
            '<div class="two-col">'
            f"<section><h3>Left: {_e(_fmt_chain(region['left_op_chain']))}</h3>{_node_table(region['left_nodes'])}</section>"
            f"<section><h3>Right: {_e(_fmt_chain(region['right_op_chain']))}</h3>{_node_table(region['right_nodes'])}</section>"
            "</div>"
            '<div class="evidence-grid">'
            f"<div><h3>Rejoin boundary</h3><p><code>{_e(region['left_anchor'])}</code> &harr; <code>{_e(region['right_anchor'])}</code></p>"
            f"<pre>{_json_pre(region['boundary_mapping'])}</pre></div>"
            f"<div><h3>Static proof evidence</h3><ul>{evidence}</ul></div>"
            "</div>"
            f"{heuristic}</div></details>"
        )

    if region_rows:
        region_table_body = "".join(region_rows)
    else:
        region_table_body = '<tr><td colspan="7" class="muted">No structural difference region was detected.</td></tr>'
    exact_table_body = "".join(exact_rows) or (
        '<tr><td colspan="5" class="muted">No ordinary exact node pairs were committed.</td></tr>'
    )

    unmatched_section = ""
    if comparison["unmatched_left_count"] or comparison["unmatched_right_count"]:
        unmatched_section = (
            '<section class="band"><div class="content"><h2>Unmatched Nodes</h2>'
            '<div class="two-col">'
            f"<section><h3>Left ({comparison['unmatched_left_count']})</h3>{_node_table(data['unmatched']['left'])}</section>"
            f"<section><h3>Right ({comparison['unmatched_right_count']})</h3>{_node_table(data['unmatched']['right'])}</section>"
            "</div></div></section>"
        )

    explanation = data["llm"]["final_explanation"]
    explanation_html = (
        f'<div class="explanation">{_e(explanation).replace(chr(10), "<br>")}</div>'
        if explanation
        else '<p class="muted">No post-match LLM explanation was requested.</p>'
    )
    call_rows = []
    for index, call in enumerate(data["llm"]["calls"], start=1):
        call_rows.append(
            "<tr>"
            f"<td>{index}</td><td><code>{_e(call.get('model'))}</code></td>"
            f"<td>{_e(call.get('context_strategy'))}</td><td>{_e(call.get('parse_ok'))}</td>"
            f"<td>{_e(call.get('prompt_tokens'))}</td><td>{_e(call.get('completion_tokens'))}</td>"
            f"<td>{_e(call.get('total_tokens'))}</td><td>{float(call.get('latency_seconds') or 0.0):.2f}s</td>"
            "</tr>"
        )
    call_table = (
        '<div class="table-wrap"><table><thead><tr><th>#</th><th>Model</th><th>Context</th><th>Parsed</th><th>Prompt</th><th>Completion</th><th>Total</th><th>Latency</th></tr></thead>'
        f"<tbody>{''.join(call_rows)}</tbody></table></div>"
        if call_rows
        else ""
    )
    config_json = _json_pre(comparison["configuration"])
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ONNX Graph Difference Report</title>
<style>
:root {{ --ink:#17212b; --muted:#607080; --line:#d9e0e6; --soft:#f5f7f9; --paper:#fff; --green:#176b45; --green-bg:#e7f5ed; --amber:#8a5a00; --amber-bg:#fff4d6; --red:#a12b2b; --red-bg:#fdeaea; --blue:#245f8f; --blue-bg:#eaf3fa; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--paper); font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }}
.content {{ width:min(1440px, calc(100% - 40px)); margin:0 auto; }}
header {{ border-bottom:1px solid var(--line); padding:28px 0 22px; }}
h1 {{ margin:0 0 8px; font-size:26px; line-height:1.25; }}
h2 {{ margin:0 0 16px; font-size:20px; }}
h3 {{ margin:0 0 10px; font-size:15px; }}
p {{ margin:8px 0; }}
.subtle,.muted {{ color:var(--muted); }}
.verdict {{ margin-top:20px; padding:16px 18px; border:1px solid var(--line); border-left:5px solid var(--blue); background:var(--soft); }}
.verdict.proven,.verdict.identical {{ border-left-color:var(--green); background:var(--green-bg); }}
.verdict.structural {{ border-left-color:var(--amber); background:var(--amber-bg); }}
.verdict.unresolved,.verdict.rejected {{ border-left-color:var(--red); background:var(--red-bg); }}
.verdict strong {{ display:block; font-size:17px; }}
.metrics {{ display:grid; grid-template-columns:repeat(6,minmax(120px,1fr)); gap:1px; margin-top:18px; border:1px solid var(--line); background:var(--line); }}
.metric {{ min-width:0; padding:12px; background:#fff; }}
.metric span {{ display:block; color:var(--muted); font-size:12px; }}
.metric strong {{ display:block; margin-top:2px; font-size:18px; overflow-wrap:anywhere; }}
.band {{ padding:24px 0; border-bottom:1px solid var(--line); }}
.band.alt {{ background:var(--soft); }}
.two-col {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:20px; }}
.model-block {{ border-top:3px solid var(--ink); padding-top:12px; min-width:0; }}
.path {{ overflow-wrap:anywhere; color:var(--muted); }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); }}
table {{ width:100%; border-collapse:collapse; background:#fff; }}
th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ background:#eef2f5; font-size:12px; color:#465563; white-space:nowrap; }}
tr:last-child td {{ border-bottom:0; }}
code {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12px; overflow-wrap:anywhere; }}
pre {{ margin:0; max-width:640px; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; }}
.badge {{ display:inline-block; padding:2px 7px; border-radius:4px; font:600 11px/1.5 system-ui,sans-serif; text-transform:uppercase; letter-spacing:0; }}
.badge.proven {{ color:var(--green); background:var(--green-bg); }}
.badge.structural,.badge.hypothesis,.badge.unverified {{ color:var(--amber); background:var(--amber-bg); }}
.badge.rejected,.badge.unresolved {{ color:var(--red); background:var(--red-bg); }}
.region {{ margin-top:14px; border:1px solid var(--line); background:#fff; }}
.region summary {{ display:flex; align-items:center; gap:12px; cursor:pointer; padding:12px 14px; background:#eef2f5; }}
.region summary span:first-child {{ font-weight:700; }}
.region summary .badge {{ margin-left:auto; }}
.region-body {{ padding:16px; }}
.evidence-grid {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:20px; margin-top:18px; }}
.note,.explanation {{ margin-top:16px; padding:13px 15px; border-left:4px solid var(--blue); background:var(--blue-bg); }}
.config {{ margin-top:14px; }}
.config summary {{ cursor:pointer; color:var(--blue); }}
footer {{ padding:20px 0 32px; color:var(--muted); font-size:12px; }}
@media (max-width:900px) {{ .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .two-col,.evidence-grid {{ grid-template-columns:1fr; }} .content {{ width:min(100% - 24px,1440px); }} }}
</style>
</head>
<body>
<header><div class="content">
  <h1>ONNX Graph Difference Report</h1>
  <p class="subtle">Static comparison without model inference. Generated {_e(data["generated_at"])}.</p>
  <div class="verdict {_e(conclusion["code"])}"><strong>{_e(conclusion["label"])}</strong><p>{_e(conclusion["detail"])}</p></div>
  <div class="metrics">
    <div class="metric"><span>Method</span><strong>{_e(comparison["method"])}</strong></div>
    <div class="metric"><span>Traversal complete</span><strong>{_e(comparison["completed"])}</strong></div>
    <div class="metric"><span>Exact pairs</span><strong>{comparison["exact_pair_count"]}</strong></div>
    <div class="metric"><span>Difference regions</span><strong>{comparison["region_count"]}</strong></div>
    <div class="metric"><span>Unmatched left / right</span><strong>{comparison["unmatched_left_count"]} / {comparison["unmatched_right_count"]}</strong></div>
    <div class="metric"><span>LLM tokens / latency</span><strong>{data["llm"]["total_tokens"]} / {data["llm"]["total_latency_seconds"]:.2f}s</strong></div>
  </div>
</div></header>

<section class="band"><div class="content"><h2>Model overview</h2><div class="two-col">
  <section class="model-block"><h3>Left graph: {_e(left["name"])}</h3><p class="path">{_e(left["path"])}</p><p>{left["node_count"]} nodes, {left["initializer_count"]} initializers, {len(left["inputs"])} inputs, {len(left["outputs"])} outputs</p><p>Opset: <code>{_e(json.dumps(left["opsets"], ensure_ascii=False))}</code></p><details><summary>Inputs / Outputs</summary><h3>Inputs</h3>{_value_table(left["inputs"])}<h3>Outputs</h3>{_value_table(left["outputs"])}</details><h3>Operator distribution</h3><div class="table-wrap"><table><thead><tr><th>Op</th><th>Count</th></tr></thead><tbody>{_histogram_table(left["op_histogram"])}</tbody></table></div></section>
  <section class="model-block"><h3>Right graph: {_e(right["name"])}</h3><p class="path">{_e(right["path"])}</p><p>{right["node_count"]} nodes, {right["initializer_count"]} initializers, {len(right["inputs"])} inputs, {len(right["outputs"])} outputs</p><p>Opset: <code>{_e(json.dumps(right["opsets"], ensure_ascii=False))}</code></p><details><summary>Inputs / Outputs</summary><h3>Inputs</h3>{_value_table(right["inputs"])}<h3>Outputs</h3>{_value_table(right["outputs"])}</details><h3>Operator distribution</h3><div class="table-wrap"><table><thead><tr><th>Op</th><th>Count</th></tr></thead><tbody>{_histogram_table(right["op_histogram"])}</tbody></table></div></section>
</div><details class="config"><summary>Comparison configuration</summary><pre>{config_json}</pre></details></div></section>

<section class="band alt"><div class="content"><h2>Difference regions summary</h2><div class="table-wrap"><table><thead><tr><th>#</th><th>Status</th><th>Rule</th><th>Left op chain</th><th>Right op chain</th><th>Rejoin anchors</th><th>Confidence</th></tr></thead><tbody>{region_table_body}</tbody></table></div></div></section>

<section class="band"><div class="content"><h2>Region evidence</h2>{"".join(region_details) or '<p class="muted">No difference regions to expand.</p>'}</div></section>
<section class="band alt"><div class="content"><h2>Exact matches, node by node</h2><p class="subtle">These node pairs do not call the LLM. Each pair passes the node-structure, existing input-mapping, constant-role and hash, and output dtype/shape checks simultaneously.</p><div class="table-wrap"><table><thead><tr><th>#</th><th>Left</th><th>Right</th><th>Why it matched</th><th>New output mapping</th></tr></thead><tbody>{exact_table_body}</tbody></table></div></div></section>
{unmatched_section}
<section class="band alt"><div class="content"><h2>LLM explanation and call statistics</h2>{explanation_html}<p class="subtle">Calls: {data["llm"]["call_count"]}; parse failures: {data["llm"]["parse_failures"]}; tokens: {data["llm"]["total_tokens"]}; latency: {data["llm"]["total_latency_seconds"]:.2f}s.</p>{call_table}</div></section>
<footer><div class="content">Report schema v{REPORT_SCHEMA_VERSION}. Mathematical equivalence is asserted only for regions marked <code>proven</code>.</div></footer>
</body>
</html>
"""


def _normalize_output_path(output: str | Path, fmt: str) -> Path:
    path = Path(output)
    desired = {"html": ".html", "markdown": ".md", "json": ".json"}[fmt]
    if path.suffix.lower() not in {".html", ".htm", ".md", ".markdown", ".json"}:
        return path.with_suffix(desired)
    return path.with_suffix(desired)


def default_report_path(left_path: str, right_path: str, *, fmt: str = "html") -> Path:
    def slug(path: str) -> str:
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(path).stem).strip("._")
        return value or "model"

    suffix = {"html": ".html", "markdown": ".md", "json": ".json", "all": ".html"}[fmt]
    return Path("reports") / f"{slug(left_path)}__vs__{slug(right_path)}{suffix}"


def write_comparison_report(
    data: dict[str, Any],
    output: str | Path,
    *,
    report_format: str = "html",
    json_sidecar: bool = True,
) -> dict[str, Path]:
    if report_format not in {"html", "markdown", "json", "all"}:
        raise ValueError(f"Unsupported report format: {report_format}")
    formats = ["html", "markdown", "json"] if report_format == "all" else [report_format]
    if json_sidecar and "json" not in formats:
        formats.append("json")

    written: dict[str, Path] = {}
    for fmt in formats:
        path = _normalize_output_path(output, fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "html":
            rendered = render_html(data)
        elif fmt == "markdown":
            rendered = render_markdown(data)
        else:
            rendered = json.dumps(data, indent=2, ensure_ascii=False)
        path.write_text(rendered, encoding="utf-8")
        written[fmt] = path.resolve()
    return written
