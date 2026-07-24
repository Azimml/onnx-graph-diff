from __future__ import annotations

from onnx_graph_diff.context import ContextConfig
from onnx_graph_diff.ir import GraphIR
from onnx_graph_diff.llm_client import LLMInvocation, QwenClient
from onnx_graph_diff.llm_matcher import LLMHeuristicMatcher, LLMMatcherConfig
from onnx_graph_diff.models import LLMCallStats, LLMMatchProposal


class ManifestClient:
    def __init__(self, regions):
        self.regions = list(regions)

    def propose_match(self, context, *, context_strategy, prompt_variant="proof_v2", feedback=None):
        expected = self.regions.pop(0)
        proposal = LLMMatchProposal(
            left_anchor=expected["left_anchor"],
            right_anchor=expected["right_anchor"],
            left_region_nodes=expected["left_region"],
            right_region_nodes=expected["right_region"],
            exit_mapping=[],
            rule_id=expected["rule_id"],
            equivalence_claim="algebraic",
            confidence=0.95,
            conditions=[],
            explanation="fixture proposal",
        )
        return LLMInvocation(
            proposal=proposal,
            stats=LLMCallStats("fake", context_strategy, parse_ok=True),
            raw_content="{}",
        )

    def explain(self, summary):
        return "fixture explanation", LLMCallStats("fake", "explanation", parse_ok=True)


def test_llm_path_recomputes_and_validates_manifest_regions(corpus_dir, corpus):
    manifest = next(item for item in corpus if item["name"] == "combined_pipeline")
    client = ManifestClient(manifest["regions"])
    matcher = LLMHeuristicMatcher(
        client,
        LLMMatcherConfig(
            context=ContextConfig(strategy="bounded", max_hops=8),
            max_retries=0,
            explain=False,
        ),
    )
    result = matcher.compare(
        GraphIR.from_path(str(corpus_dir / manifest["left"])),
        GraphIR.from_path(str(corpus_dir / manifest["right"])),
    )
    assert result.completed
    assert len([event for event in result.events if event.kind == "region"]) == 3
    assert len(result.llm_calls) == 3


def test_json_extraction_accepts_fenced_thinking_output():
    parsed = QwenClient._extract_json('analysis\n```json\n{"left_anchor":"a"}\n```')
    assert parsed["left_anchor"] == "a"


def test_exit_mapping_accepts_object_or_pair_array():
    assert QwenClient._normalize_exit_mapping({"left": "right"}) == [("left", "right")]
    assert QwenClient._normalize_exit_mapping([["left", "right"]]) == [("left", "right")]
