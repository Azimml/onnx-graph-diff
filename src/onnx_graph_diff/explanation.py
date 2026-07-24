"""Post-match natural-language explanation of a comparison result.

:func:`build_explanation_summary` distils a finished
:class:`~onnx_graph_diff.models.ComparisonResult` down to just its difference
regions -- compact node descriptors, anchors, proof status, validated rule, and
any per-region heuristic note -- so the model is asked to explain what the
independent checker already decided rather than to re-derive it.
:func:`explain_result` then calls an :class:`ExplanationClient`, attaches the
prose to ``result.explanation``, and records the call's
:class:`~onnx_graph_diff.models.LLMCallStats`.

The client is a narrow :class:`~typing.Protocol`, so the routes and the tests
can supply a real LLM client or a fake interchangeably.
"""

from __future__ import annotations

from typing import Any, Protocol

from .ir import GraphIR
from .models import ComparisonResult, LLMCallStats


class ExplanationClient(Protocol):
    def explain(self, summary: dict[str, Any]) -> tuple[str, LLMCallStats]: ...


def build_explanation_summary(left: GraphIR, right: GraphIR, result: ComparisonResult) -> dict[str, Any]:
    regions: list[dict[str, Any]] = []
    for event in result.events:
        if event.kind != "region":
            continue
        regions.append(
            {
                "left_nodes": [left.compact_node(node_id) for node_id in event.left_nodes],
                "right_nodes": [right.compact_node(node_id) for node_id in event.right_nodes],
                "anchors": [event.left_anchor, event.right_anchor],
                "proof_status": event.proof.status if event.proof else None,
                "validated_rule": event.proof.rule_id if event.proof else None,
                "proof_details": event.proof.details if event.proof else [],
                "heuristic_explanation": event.explanation,
            }
        )
    return {
        "method": result.method,
        "completed": result.completed,
        "regions": regions,
        "unmatched_left": result.unmatched_left,
        "unmatched_right": result.unmatched_right,
    }


def explain_result(
    client: ExplanationClient,
    left: GraphIR,
    right: GraphIR,
    result: ComparisonResult,
) -> ComparisonResult:
    explanation, stats = client.explain(build_explanation_summary(left, right, result))
    result.explanation = explanation
    result.llm_calls.append(stats)
    return result
