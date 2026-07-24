from __future__ import annotations

from onnx_graph_diff.agentic_matcher import AgenticMatcherConfig, LLMAgenticMatcher
from onnx_graph_diff.evaluation import score_result
from onnx_graph_diff.ir import GraphIR
from onnx_graph_diff.llm_client import AgenticLLMInvocation, QwenClient
from onnx_graph_diff.models import AgenticMatchProposal, LLMCallStats


class SequenceAgentClient:
    def __init__(self, actions):
        self.actions = list(actions)
        self.contexts = []
        self.feedback = []

    def propose_agentic_match(self, context, *, context_strategy, prompt_variant="agent_v3", feedback=None):
        self.contexts.append(context)
        self.feedback.append(feedback)
        action = self.actions.pop(0)
        if action == "expand":
            proposal = AgenticMatchProposal(
                action="expand",
                expand_hops=2,
                expand_nodes=6,
                explanation="need a wider downstream window",
            )
        else:
            proposal = AgenticMatchProposal(
                action="match",
                left_anchor=action["left_anchor"],
                right_anchor=action["right_anchor"],
                left_region_nodes=action.get("left_region", []),
                right_region_nodes=action.get("right_region", []),
                rule_id=action.get("rule_id"),
                equivalence_claim="algebraic",
                confidence=0.9,
                explanation="agent-selected rejoin",
                same_op_pairs_seen=[(action["left_anchor"], action["right_anchor"])],
            )
        return AgenticLLMInvocation(
            proposal=proposal,
            stats=LLMCallStats("fake-agent", context_strategy, parse_ok=True),
            raw_content="{}",
        )

    def explain(self, summary):
        return "agent explanation", LLMCallStats("fake-agent", "explanation", parse_ok=True)


def _regions(manifest):
    return manifest.get(
        "regions",
        [
            {
                "left_anchor": manifest["left_anchor"],
                "right_anchor": manifest["right_anchor"],
                "left_region": manifest["left_region"],
                "right_region": manifest["right_region"],
                "rule_id": manifest["rule_id"],
            }
        ],
    )


def test_agentic_route_expands_without_receiving_candidates(corpus_dir, corpus):
    manifest = next(item for item in corpus if item["name"] == "combined_pipeline")
    client = SequenceAgentClient(["expand", *_regions(manifest)])
    matcher = LLMAgenticMatcher(
        client,
        AgenticMatcherConfig(
            initial_hops=2,
            initial_nodes_per_graph=4,
            max_hops=8,
            max_nodes_per_graph=20,
            max_expansions=2,
            max_retries_per_mismatch=1,
            max_calls_per_mismatch=6,
        ),
    )
    result = matcher.compare(
        GraphIR.from_path(str(corpus_dir / manifest["left"])),
        GraphIR.from_path(str(corpus_dir / manifest["right"])),
    )
    assert result.completed
    assert result.method == "llm_agent"
    assert len([event for event in result.events if event.kind == "region"]) == 3
    assert result.metadata["total_expansions"] == 1
    assert "coarse_anchor_pairs" not in client.contexts[0]
    assert client.contexts[0]["agent_control"]["candidate_pairs_are_not_supplied"] is True
    assert (
        client.contexts[1]["agent_control"]["current_max_hops"]
        > client.contexts[0]["agent_control"]["current_max_hops"]
    )
    final_context = client.contexts[-1]
    assert "left_anchor_2" in final_context["left"]["visible_node_ids"]
    assert "left_anchor_2" in final_context["left"]["visible_matched_history_ids"]
    assert "left_anchor_2" not in final_context["left"]["visible_anchor_ids"]
    assert "left_anchor_3" in final_context["left"]["visible_anchor_ids"]


def test_agentic_route_rejects_bad_anchor_and_returns_feedback(corpus_dir, corpus):
    manifest = next(item for item in corpus if item["name"] == "identity_elimination")
    bad = {"left_anchor": "left_identity", "right_anchor": manifest["right_anchor"]}
    client = SequenceAgentClient([bad, _regions(manifest)[0]])
    matcher = LLMAgenticMatcher(
        client,
        AgenticMatcherConfig(
            initial_hops=2,
            initial_nodes_per_graph=8,
            max_hops=8,
            max_nodes_per_graph=32,
            max_expansions=3,
            max_retries_per_mismatch=2,
            max_calls_per_mismatch=3,
        ),
    )
    result = matcher.compare(
        GraphIR.from_path(str(corpus_dir / manifest["left"])),
        GraphIR.from_path(str(corpus_dir / manifest["right"])),
    )
    assert result.completed
    assert client.feedback[1]
    assert "different domain/op types" in client.feedback[1]
    assert client.contexts[0]["agent_control"]["can_expand"] is False
    assert client.contexts[0]["agent_control"]["hidden_downstream_detected"] is False
    assert result.llm_calls[0].metadata["accepted"] is False
    assert result.llm_calls[1].metadata["accepted"] is True


def test_agentic_route_can_select_visible_virtual_output(corpus_dir, corpus):
    manifest = next(item for item in corpus if item["name"] == "output_matmul_gemm")
    client = SequenceAgentClient([_regions(manifest)[0]])
    matcher = LLMAgenticMatcher(
        client,
        AgenticMatcherConfig(
            initial_hops=8,
            initial_nodes_per_graph=20,
            max_hops=8,
            max_nodes_per_graph=20,
            max_expansions=0,
        ),
    )
    result = matcher.compare(
        GraphIR.from_path(str(corpus_dir / manifest["left"])),
        GraphIR.from_path(str(corpus_dir / manifest["right"])),
    )
    assert result.completed
    assert result.events[0].left_anchor == "__graph_output__:0"
    assert "__graph_output__:0" in client.contexts[0]["left"]["visible_anchor_ids"]


class ManifestWindowAgent:
    def __init__(self, manifest):
        self.expected = list(_regions(manifest))
        self.equivalent = manifest.get("equivalent", True)

    def propose_agentic_match(self, context, *, context_strategy, prompt_variant="agent_v3", feedback=None):
        if not self.equivalent:
            proposal = AgenticMatchProposal(action="unresolved", explanation="known negative fixture")
        else:
            expected = self.expected[0]
            visible_left = set(context["left"]["visible_anchor_ids"])
            visible_right = set(context["right"]["visible_anchor_ids"])
            if expected["left_anchor"] not in visible_left or expected["right_anchor"] not in visible_right:
                proposal = AgenticMatchProposal(
                    action="expand",
                    expand_hops=2,
                    expand_nodes=8,
                    explanation="expected rejoin is beyond the current window",
                )
            else:
                self.expected.pop(0)
                proposal = AgenticMatchProposal(
                    action="match",
                    left_anchor=expected["left_anchor"],
                    right_anchor=expected["right_anchor"],
                    left_region_nodes=expected["left_region"],
                    right_region_nodes=expected["right_region"],
                    rule_id=expected["rule_id"],
                    equivalence_claim="algebraic",
                    confidence=1.0,
                    same_op_pairs_seen=[(expected["left_anchor"], expected["right_anchor"])],
                )
        return AgenticLLMInvocation(
            proposal=proposal,
            stats=LLMCallStats("manifest-agent", context_strategy, parse_ok=True),
            raw_content="{}",
        )

    def explain(self, summary):
        return "", LLMCallStats("manifest-agent", "explanation", parse_ok=True)


def test_agentic_window_protocol_covers_controlled_corpus(corpus_dir, corpus):
    records = []
    for manifest in corpus:
        matcher = LLMAgenticMatcher(
            ManifestWindowAgent(manifest),
            AgenticMatcherConfig(
                initial_hops=1,
                initial_nodes_per_graph=4,
                max_hops=8,
                max_nodes_per_graph=24,
                max_expansions=4,
                max_calls_per_mismatch=8,
            ),
        )
        result = matcher.compare(
            GraphIR.from_path(str(corpus_dir / manifest["left"])),
            GraphIR.from_path(str(corpus_dir / manifest["right"])),
        )
        records.append(score_result(manifest, result))
    assert len(records) == 10
    assert all(record["outcome_correct"] for record in records)
    assert not any(record["false_positive"] for record in records)


def test_agentic_parser_normalizes_non_integer_expansion_fields():
    assert QwenClient._normalize_expand_amount([], default=8, maximum=16) == 8
    assert QwenClient._normalize_expand_amount("4", default=2, maximum=4) == 4
    assert QwenClient._normalize_expand_amount(0, default=2, maximum=4) == 1
    assert QwenClient._normalize_anchor_id({"id": "left_anchor", "op": "Relu"}) == "left_anchor"
    assert QwenClient._normalize_anchor_id("right_anchor") == "right_anchor"
    assert QwenClient._normalize_id_pairs([["left_relu", "right_relu"]]) == [("left_relu", "right_relu")]
    assert QwenClient._normalize_id_pairs([{"left_anchor": {"id": "left_add"}, "right_anchor": "right_add"}]) == [
        ("left_add", "right_add")
    ]
    assert QwenClient._normalize_id_pairs('["left_relu", "right_relu"]') == [("left_relu", "right_relu")]
    assert QwenClient._normalize_id_pairs('[["left_relu", "right_relu"], ["left_add", "right_add"]]') == [
        ("left_relu", "right_relu"),
        ("left_add", "right_add"),
    ]
    assert QwenClient._normalize_conditions({"can_expand": True}) == ["can_expand=True"]
