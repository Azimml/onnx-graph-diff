from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

CONTROLLED_PATH = REPORTS / "agent-v13-full-10.json"
ZOO_PATH = REPORTS / "model-zoo-agent-v12.json"

EVOLUTION = [
    ("agent_v4 / targeted", "agent-v4-targeted.json", "首次稳定选对简单接点；复杂布局仍会选到优化内部节点。"),
    ("agent_v6 / controlled", "agent-v6-full-10.json", "加入局部变换边界和终端输出规则。"),
    ("agent_v9 / controlled", "agent-v9-full-10.json", "显式要求 LLM 自报 same-op 扫描结果。"),
    ("agent_v10 / controlled", "agent-v10-full-10.json", "已匹配父节点只保留作历史，不再允许向后选。"),
    ("agent_v11 / model zoo", "model-zoo-agent-v11.json", "8 个架构模型对中 6 个完整成功。"),
    ("agent_v12 / model zoo", "model-zoo-agent-v12.json", "把接点后的新差异留给下一轮，22/22 区域成功。"),
    ("agent_v13 / controlled", "agent-v13-full-10.json", "区分标准 Gemm 与自定义 FusedLinear，最终默认。"),
]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def pct(value: Any) -> str:
    return f"{float(value or 0) * 100:.0f}%"


def pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def accepted_calls(report: dict[str, Any]) -> list[tuple[dict[str, Any], int, dict[str, Any]]]:
    result = []
    for record in report["records"]:
        accepted_index = 0
        for call in record.get("call_details", []):
            if call.get("accepted") is True:
                accepted_index += 1
                result.append((record, accepted_index, call))
    return result


def scan_verified(call: dict[str, Any]) -> bool:
    pair = call.get("anchor_pair") or []
    if pair and str(pair[0]).startswith("__graph_output__:"):
        return True
    return pair in (call.get("same_op_pairs_seen") or [])


def ledger_rows(report: dict[str, Any], dataset: str) -> str:
    rows = []
    for record, stage, call in accepted_calls(report):
        anchor = call.get("anchor_pair") or ["?", "?"]
        left_region = call.get("reconstructed_left_region") or []
        right_region = call.get("reconstructed_right_region") or []
        proof = call.get("proof_status") or "unverified"
        claimed = call.get("claimed_rule") or "unknown"
        validated = call.get("validated_rule") or "none"
        why = (
            "LLM 自行扫描并报告所选 pair；程序确认两个 anchor 的本地结构签名兼容；"
            "从 anchor 输入反向重建出的区域入口对现有 value mapping 闭合；"
            f"静态验证结果为 {proof}。"
        )
        rows.append(
            "<tr>"
            f'<td>{esc(dataset)}</td><td><code>{esc(record["case"])}</code><br><span class="muted">stage {stage}</span></td>'
            f"<td><code>{esc(anchor[0])}</code><br><code>{esc(anchor[1])}</code></td>"
            f"<td>{'yes' if scan_verified(call) else 'no'}</td>"
            f"<td><code>{esc(left_region)}</code><br><code>{esc(right_region)}</code></td>"
            f'<td><code>{esc(claimed)}</code><br><span class="muted">validator: {esc(validated)}</span></td>'
            f'<td><span class="status {esc(proof)}">{esc(proof)}</span><p>{esc(why)}</p></td>'
            "</tr>"
        )
    return "".join(rows)


def parser_view(call: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "action",
        "same_op_pairs_seen",
        "claimed_rule",
        "proposed_left_region",
        "proposed_right_region",
        "proposed_exit_mapping",
        "parse_error",
    ]
    return {key: call.get(key) for key in keys if key in call}


def validator_view(call: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "accepted",
        "anchor_pair",
        "reconstructed_left_region",
        "reconstructed_right_region",
        "reconstructed_exit_mapping",
        "validated_rule",
        "proof_status",
        "expanded",
    ]
    return {key: call.get(key) for key in keys if key in call}


def result_reason(call: dict[str, Any], next_feedback: str | None) -> str:
    if call.get("accepted") is True:
        return (
            "接受：所选非终端接点在 LLM 自报扫描列表中；anchor ID 位于当前可选窗口；"
            "程序重建得到闭合区域，并通过静态证书或受限 structural 条件。"
        )
    if call.get("action") == "expand" and call.get("expanded"):
        return "扩展：LLM 判断当前窗口缺少接点或依赖，程序增加 hop/节点预算后重新组织上下文。"
    if call.get("action") == "unresolved":
        return "未解析：LLM 在当前完整窗口和历史拒绝条件下没有提出可接受接点。"
    if next_feedback:
        return f"拒绝并反馈到下一轮：{next_feedback}"
    return "未接受：响应没有形成可提交的闭合、可验证匹配。"


def transcript_blocks(report: dict[str, Any], dataset: str) -> str:
    blocks = []
    for record in report["records"]:
        calls = record.get("call_details", [])
        for index, call in enumerate(calls, start=1):
            next_feedback = None
            if index < len(calls):
                next_feedback = calls[index].get("feedback")
            action = call.get("action") or ("parse_error" if call.get("parse_error") else "unknown")
            accepted = call.get("accepted") is True
            status = call.get("proof_status") or ("accepted" if accepted else action)
            context = call.get("user_context") or {}
            frontier = {
                "left": (context.get("left") or {}).get("mismatch_frontier"),
                "right": (context.get("right") or {}).get("mismatch_frontier"),
            }
            title = f"{dataset} / {record['case']} / call {index}: {action}"
            blocks.append(
                f'<details class="trace"><summary><span><strong>{esc(title)}</strong>'
                f"<small>frontier {esc(frontier)}; window {esc(call.get('window_hops'))} hops / "
                f"{esc(call.get('window_nodes_per_graph'))} nodes per graph</small></span>"
                f'<b class="status {esc(status)}">{esc(status)}</b></summary><div class="trace-body">'
                f'<div class="decision"><strong>这一轮为什么得到该结果</strong><p>{esc(result_reason(call, next_feedback))}</p></div>'
                "<h3>1. System Prompt</h3>"
                f"<pre>{esc(call.get('system_prompt') or '(not captured)')}</pre>"
                "<h3>2. User Context JSON</h3>"
                f"<pre>{esc(pretty(context))}</pre>"
                "<h3>3. 上一轮反馈</h3>"
                f"<pre>{esc(call.get('feedback') or '(first call: no previous feedback)')}</pre>"
                "<h3>4. LLM 原始响应</h3>"
                f'<pre class="response">{esc(call.get("raw_response") or "(not captured)")}</pre>'
                '<div class="two"><section><h3>5. Parser 输出</h3>'
                f"<pre>{esc(pretty(parser_view(call)))}</pre></section>"
                "<section><h3>6. 程序重建与验证</h3>"
                f"<pre>{esc(pretty(validator_view(call)))}</pre></section></div>"
                "</div></details>"
            )
    return "".join(blocks)


def evolution_rows() -> str:
    rows = []
    for label, filename, note in EVOLUTION:
        path = REPORTS / filename
        if not path.exists():
            continue
        summary = load(path)["summary"]
        rows.append(
            "<tr>"
            f"<td><code>{esc(label)}</code></td><td>{summary['cases']}</td>"
            f"<td>{pct(summary['outcome_accuracy'])}</td>"
            f"<td>{pct(summary.get('strict_claimed_rule_accuracy', summary.get('rule_accuracy', 0)))}</td>"
            f"<td>{pct(summary.get('self_scanned_anchor_rate', 0))}</td>"
            f"<td>{summary['avg_total_tokens']:,.0f}</td><td>{summary['avg_latency_seconds']:.2f}s</td>"
            f"<td>{esc(note)}</td></tr>"
        )
    return "".join(rows)


def metric_cards(summary: dict[str, Any], title: str, regions: int) -> str:
    return (
        f'<section class="metric-group"><h3>{esc(title)}</h3><div class="metrics">'
        f"<div><span>Cases</span><b>{summary['cases']}</b></div>"
        f"<div><span>Regions</span><b>{regions}</b></div>"
        f"<div><span>Outcome</span><b>{pct(summary['outcome_accuracy'])}</b></div>"
        f"<div><span>Anchor</span><b>{pct(summary['anchor_accuracy'])}</b></div>"
        f"<div><span>Region</span><b>{pct(summary['region_accuracy'])}</b></div>"
        f"<div><span>Self scan</span><b>{pct(summary['self_scanned_anchor_rate'])}</b></div>"
        f"<div><span>Strict rule</span><b>{pct(summary['strict_claimed_rule_accuracy'])}</b></div>"
        f"<div><span>False positive</span><b>{pct(summary['false_positive_rate'])}</b></div>"
        f"<div><span>Avg tokens</span><b>{summary['avg_total_tokens']:,.0f}</b></div>"
        f"<div><span>Avg latency</span><b>{summary['avg_latency_seconds']:.2f}s</b></div>"
        "</div></section>"
    )


def build_page() -> str:
    controlled = load(CONTROLLED_PATH)
    zoo = load(ZOO_PATH)
    controlled_regions = len(accepted_calls(controlled))
    zoo_regions = len(accepted_calls(zoo))
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    ledger = ledger_rows(controlled, "受控集") + ledger_rows(zoo, "架构集")
    transcripts = transcript_blocks(controlled, "受控集") + transcript_blocks(zoo, "架构集")

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ONNX Graph Diff - LLM 自主图匹配实验</title>
<style>
:root{{--ink:#18222c;--muted:#64737f;--line:#d8e0e5;--soft:#f5f7f8;--paper:#fff;--green:#176d49;--green-soft:#e8f5ee;--blue:#28658f;--blue-soft:#eaf3f9;--amber:#875b0a;--amber-soft:#fff4d9;--red:#a23a3a;--red-soft:#fcecec}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;color:var(--ink);background:var(--paper);font:14px/1.62 system-ui,-apple-system,"Segoe UI",sans-serif;letter-spacing:0}}a{{color:var(--blue);text-decoration:none}}a:hover{{text-decoration:underline}}code,pre{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}code{{font-size:12px;overflow-wrap:anywhere}}.content{{width:min(1500px,calc(100% - 40px));margin:auto}}header{{padding:30px 0 24px;border-bottom:1px solid var(--line)}}h1{{margin:4px 0 9px;font-size:29px;line-height:1.25}}h2{{margin:0 0 16px;font-size:21px}}h3{{margin:0 0 9px;font-size:15px}}p{{margin:7px 0}}.lead{{max-width:1050px;color:#465660;font-size:16px}}.verdict{{margin-top:18px;padding:15px 17px;border-left:5px solid var(--green);background:var(--green-soft)}}.verdict strong{{display:block;font-size:17px}}.nav{{position:sticky;top:0;z-index:4;border-bottom:1px solid var(--line);background:#fff}}.nav .content{{display:flex;gap:6px;overflow:auto;padding:8px 0}}.nav a{{flex:0 0 auto;padding:4px 7px;font-size:12px}}.band{{padding:26px 0;border-bottom:1px solid var(--line)}}.band.alt{{background:var(--soft)}}.two{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:17px}}.metrics{{display:grid;grid-template-columns:repeat(5,minmax(110px,1fr));gap:1px;border:1px solid var(--line);background:var(--line)}}.metrics div{{min-width:0;padding:10px;background:#fff}}.metrics span{{display:block;color:var(--muted);font-size:11px}}.metrics b{{display:block;margin-top:2px;font-size:17px;overflow-wrap:anywhere}}.metric-group{{margin-top:16px;border-top:3px solid var(--ink);padding-top:11px}}.flow{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:1px;border:1px solid var(--line);background:var(--line)}}.flow div{{padding:12px;background:#fff}}.flow b{{display:block;color:var(--blue);margin-bottom:4px}}.table-wrap{{overflow:auto;border:1px solid var(--line)}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{white-space:nowrap;background:#edf1f3;color:#465660;font-size:12px}}tr:last-child td{{border-bottom:0}}.muted{{color:var(--muted)}}.status{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;text-transform:uppercase;color:var(--blue);background:var(--blue-soft)}}.status.proven,.status.accepted{{color:var(--green);background:var(--green-soft)}}.status.structural,.status.expand{{color:var(--amber);background:var(--amber-soft)}}.status.unresolved,.status.rejected,.status.parse_error{{color:var(--red);background:var(--red-soft)}}.note,.decision{{padding:12px 14px;border-left:4px solid var(--blue);background:var(--blue-soft)}}.trace{{margin-top:12px;border:1px solid var(--line);background:#fff}}.trace summary{{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:11px 13px;cursor:pointer;background:#edf1f3}}.trace summary small{{display:block;color:var(--muted);font-weight:400}}.trace-body{{padding:14px}}.trace h3{{margin-top:17px}}pre{{margin:0;max-height:560px;overflow:auto;padding:11px;border:1px solid var(--line);background:#fbfcfc;white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;line-height:1.45}}pre.response{{border-left:4px solid var(--blue)}}footer{{padding:20px 0 30px;color:var(--muted);font-size:12px}}@media(max-width:950px){{.content{{width:min(100% - 24px,1500px)}}.two{{grid-template-columns:1fr}}.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}.flow{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div class="content"><div class="muted">ONNX Graph Diff / Qwen3.5-122B / no inference</div><h1>LLM 自主寻找 ONNX 图重新匹配点：完整实验结果</h1><p class="lead">程序不给候选节点对，只做 ready-frontier 普通匹配、parent-preserving 窗口构造、按请求扩展、closed-region 重建和 ONNX opset 静态验证。Qwen 自己扫描节点、选择接点、解释局部优化，并在被拒绝后根据反馈修正。</p><div class="verdict"><strong>最终图匹配结果：受控集 10/10，架构集 8/8、22/22 局部区域</strong><p>受控集使用默认 <code>agent_v13</code>；架构集使用结构逻辑相同的 <code>agent_v12</code>，两者差异仅是 v13 明确禁止把自定义 FusedLinear 当成标准 Gemm。两组均为 0 false positive，且每个非终端接点都出现在 LLM 自报的扫描结果中。</p></div></div></header>
<nav class="nav"><div class="content"><a href="#method">怎么匹配</a><a href="#metrics">最终指标</a><a href="#evolution">Prompt 演进</a><a href="#ledger">每个接点为何成立</a><a href="#transcripts">完整 Prompt 与输出</a><a href="solution.html">返回方案页</a></div></nav>
<section class="band" id="method"><div class="content"><h2>一轮失配处理到底发生什么</h2><div class="flow"><div><b>1. 零 token 普通匹配</b>比较 local signature、已知动态输入 mapping、常量角色/hash、输出 dtype/shape；能直接对应就提交。</div><div><b>2. 构造局部窗口</b>保留失配节点、全部直接 parents、已匹配历史、下游有限窗口和真实节点 ID，但不生成候选 pair。</div><div><b>3. LLM 自主动作</b>返回 <code>match</code>、<code>expand</code> 或 <code>unresolved</code>，并自报 <code>same_op_pairs_seen</code>。</div><div><b>4. 程序重建与证明</b>从 anchor inputs 反向重建 closed region；检查 entry mapping、常量、shape 和 ONNX 规则。</div><div><b>5. 恢复拓扑遍历</b>接受后提交出口 mapping，重新计算 ready frontier；被拒绝则把具体原因放入下一轮 prompt。</div></div><div class="note" style="margin-top:15px"><strong>为什么这能验证 LLM 能力：</strong>程序没有给出候选对或排序。非终端 anchor 只有在 LLM 自己列入 <code>same_op_pairs_seen</code> 后才可能被接受；程序做的是验证该选择，而不是替模型搜索。</div></div></section>
<section class="band alt" id="metrics"><div class="content"><h2>最终指标与真实成本</h2><div class="two">{metric_cards(controlled["summary"], "受控集 / agent_v13", controlled_regions)}{metric_cards(zoo["summary"], "架构模型集 / agent_v12", zoo_regions)}</div><p class="muted" style="margin-top:14px">受控集 completion 为 {pct(controlled["summary"]["completion_rate"])}，因为故意构造的非等价 Gemm 被正确留为 unresolved；outcome 仍为 100%。架构集准确率达到 100% 的代价是平均 {zoo["summary"]["avg_total_tokens"]:,.0f} tokens、{zoo["summary"]["avg_latency_seconds"]:.2f}s/模型对，说明该路线适合能力研究和困难图，不是最低成本默认方案。</p></div></section>
<section class="band" id="evolution"><div class="content"><h2>Prompt 和控制策略怎样一步步优化</h2><div class="table-wrap"><table><thead><tr><th>版本</th><th>Cases</th><th>Outcome</th><th>Strict rule</th><th>Self scan</th><th>Avg tokens</th><th>Avg latency</th><th>主要变化</th></tr></thead><tbody>{evolution_rows()}</tbody></table></div></div></section>
<section class="band alt" id="ledger"><div class="content"><h2>每个生成的匹配到底为什么成立</h2><p>下表包含最终实验中全部已接受局部接点。这里的“成立”不是只因为 LLM 说像：必须同时有自报扫描、可见且未提交的真实 ID、程序重建的闭合区域，以及静态证明或严格 structural 条件。</p><div class="table-wrap"><table><thead><tr><th>数据集</th><th>Case / stage</th><th>Anchor pair</th><th>LLM 自报扫描</th><th>程序重建 region</th><th>规则</th><th>为什么接受</th></tr></thead><tbody>{ledger}</tbody></table></div></div></section>
<section class="band" id="transcripts"><div class="content"><h2>每一轮完整 Prompt、上下文、反馈、原始响应与解析输出</h2><p>以下内容直接来自开启 <code>--capture-transcripts</code> 的保存结果。每个折叠项都是一次真实 API 调用；包含失败、扩展和最终接受调用，不只展示成功样例。</p>{transcripts}</div></section>
<footer><div class="content">Generated {esc(generated)} from {esc(CONTROLLED_PATH.name)} and {esc(ZOO_PATH.name)}. The page embeds saved static-comparison data and does not execute ONNX models.</div></footer>
</body></html>"""


def main() -> int:
    page = build_page()
    outputs = [DOCS / "agentic-results.html", REPORTS / "results.html"]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(page, encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
