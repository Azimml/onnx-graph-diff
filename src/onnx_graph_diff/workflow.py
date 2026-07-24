"""LangGraph state machine wiring load -> route -> match -> explain.

:func:`build_workflow` compiles a small :class:`~langgraph.graph.StateGraph`
over :class:`WorkflowState`: it loads both models into a
:class:`~onnx_graph_diff.ir.GraphIR`, dispatches on ``method`` to the
fingerprint, LLM, or agentic matcher, and -- when ``explain`` is set and an LLM
client is available -- runs a post-match natural-language explanation pass.

The matchers' own ``explain`` flags are forced off so explanation happens once,
in a single dedicated node, rather than inside each route. The compiled graph is
invoked with a dict of paths/method/explain and returns the populated state
(including ``left_graph``, ``right_graph``, and ``result``).
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .agentic_matcher import AgenticMatcherConfig, LLMAgenticMatcher
from .explanation import explain_result
from .ir import GraphIR
from .llm_matcher import LLMHeuristicMatcher, LLMMatcherConfig, MatchClient
from .matcher import FingerprintMatcher, MatcherConfig
from .models import ComparisonResult


class WorkflowState(TypedDict, total=False):
    left_path: str
    right_path: str
    method: Literal["fingerprint", "llm", "llm_agent"]
    explain: bool
    left_graph: GraphIR
    right_graph: GraphIR
    result: ComparisonResult


def build_workflow(
    *,
    llm_client: MatchClient | None = None,
    fingerprint_config: MatcherConfig | None = None,
    llm_config: LLMMatcherConfig | None = None,
    agentic_config: AgenticMatcherConfig | None = None,
):
    llm_config = llm_config or LLMMatcherConfig(explain=False)
    llm_config.explain = False
    agentic_config = agentic_config or AgenticMatcherConfig(explain=False)
    agentic_config.explain = False

    def load_graphs(state: WorkflowState) -> WorkflowState:
        return {
            "left_graph": GraphIR.from_path(state["left_path"]),
            "right_graph": GraphIR.from_path(state["right_path"]),
        }

    def route(state: WorkflowState) -> str:
        return state["method"]

    def fingerprint_match(state: WorkflowState) -> WorkflowState:
        result = FingerprintMatcher(fingerprint_config).compare(state["left_graph"], state["right_graph"])
        return {"result": result}

    def llm_match(state: WorkflowState) -> WorkflowState:
        if llm_client is None:
            raise RuntimeError("llm_client is required for the LLM workflow path")
        result = LLMHeuristicMatcher(llm_client, llm_config).compare(state["left_graph"], state["right_graph"])
        return {"result": result}

    def llm_agent_match(state: WorkflowState) -> WorkflowState:
        if llm_client is None:
            raise RuntimeError("llm_client is required for the LLM-agent workflow path")
        result = LLMAgenticMatcher(llm_client, agentic_config).compare(state["left_graph"], state["right_graph"])
        return {"result": result}

    def explain(state: WorkflowState) -> WorkflowState:
        if not state.get("explain") or llm_client is None:
            return {}
        return {
            "result": explain_result(
                llm_client,
                state["left_graph"],
                state["right_graph"],
                state["result"],
            )
        }

    builder = StateGraph(WorkflowState)
    builder.add_node("load_graphs", load_graphs)
    builder.add_node("fingerprint", fingerprint_match)
    builder.add_node("llm", llm_match)
    builder.add_node("llm_agent", llm_agent_match)
    builder.add_node("explain", explain)
    builder.add_edge(START, "load_graphs")
    builder.add_conditional_edges(
        "load_graphs",
        route,
        {"fingerprint": "fingerprint", "llm": "llm", "llm_agent": "llm_agent"},
    )
    builder.add_edge("fingerprint", "explain")
    builder.add_edge("llm", "explain")
    builder.add_edge("llm_agent", "explain")
    builder.add_edge("explain", END)
    return builder.compile()
