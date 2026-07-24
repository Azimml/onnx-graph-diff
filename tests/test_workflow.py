from __future__ import annotations

from onnx_graph_diff.agentic_matcher import AgenticMatcherConfig
from onnx_graph_diff.llm_client import AgenticLLMInvocation
from onnx_graph_diff.models import AgenticMatchProposal, LLMCallStats
from onnx_graph_diff.workflow import build_workflow


def test_fingerprint_langgraph_workflow(corpus_dir):
    workflow = build_workflow()
    state = workflow.invoke(
        {
            "left_path": str(corpus_dir / "matmul_add_gemm.left.onnx"),
            "right_path": str(corpus_dir / "matmul_add_gemm.right.onnx"),
            "method": "fingerprint",
            "explain": False,
        }
    )
    assert state["result"].completed
    assert state["result"].method == "fingerprint"


class _AgentClient:
    def propose_agentic_match(self, context, *, context_strategy, prompt_variant="agent_v13", feedback=None):
        proposal = AgenticMatchProposal(
            action="match",
            left_anchor="left_relu",
            right_anchor="right_relu",
            left_region_nodes=["left_matmul", "left_add"],
            right_region_nodes=["right_gemm"],
            rule_id="matmul_add_to_gemm",
            equivalence_claim="algebraic",
            confidence=1.0,
            same_op_pairs_seen=[("left_relu", "right_relu")],
        )
        return AgenticLLMInvocation(
            proposal=proposal,
            stats=LLMCallStats("fake-agent", context_strategy, parse_ok=True),
            raw_content="{}",
        )

    def explain(self, summary):
        return "", LLMCallStats("fake-agent", "explanation", parse_ok=True)


def test_llm_agent_langgraph_workflow(corpus_dir):
    workflow = build_workflow(
        llm_client=_AgentClient(),
        agentic_config=AgenticMatcherConfig(explain=False),
    )
    state = workflow.invoke(
        {
            "left_path": str(corpus_dir / "matmul_add_gemm.left.onnx"),
            "right_path": str(corpus_dir / "matmul_add_gemm.right.onnx"),
            "method": "llm_agent",
            "explain": False,
        }
    )
    assert state["result"].completed
    assert state["result"].method == "llm_agent"
