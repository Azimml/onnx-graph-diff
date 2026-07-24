from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from .models import AgenticMatchProposal, LLMCallStats, LLMMatchProposal
from .providers import (
    DEFAULT_PROVIDER,
    ProviderProfile,
    profile_with_overrides,
    resolve_provider,
)

# Backwards-compatible defaults. The DashScope/Qwen backend remains the default
# provider, so these constants keep pointing at it; provider-specific defaults
# now live in ``providers.ProviderProfile``.
DEFAULT_MODEL = resolve_provider(DEFAULT_PROVIDER).default_model
DEFAULT_BASE_URL = resolve_provider(DEFAULT_PROVIDER).default_base_url


class _ProposalPayload(BaseModel):
    left_anchor: str | None = None
    right_anchor: str | None = None
    left_region_nodes: list[str] = Field(default_factory=list)
    right_region_nodes: list[str] = Field(default_factory=list)
    exit_mapping: Any = Field(default_factory=list)
    rule_id: str | None = None
    equivalence_claim: str = "unknown"
    confidence: float = 0.0
    conditions: list[str] = Field(default_factory=list)
    explanation: str = ""


class _AgenticProposalPayload(_ProposalPayload):
    action: str = "unresolved"
    left_anchor: Any = None
    right_anchor: Any = None
    equivalence_claim: Any = "unknown"
    expand_hops: Any = 2
    expand_nodes: Any = 8
    same_op_pairs_seen: Any = Field(default_factory=list)
    conditions: Any = Field(default_factory=list)


@dataclass
class LLMConfig:
    """Provider-agnostic configuration for the chat client.

    ``provider`` selects a backend profile (see :mod:`onnx_graph_diff.providers`).
    ``model`` and ``base_url`` are optional overrides -- when left as ``None``
    they fall back to the selected provider's defaults, so the common case only
    needs to name a provider (or nothing at all for DashScope/Qwen).
    """

    provider: str = DEFAULT_PROVIDER
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    #: Override the environment variable the API key is read from. Useful for the
    #: generic ``openai-compatible`` provider pointed at a custom gateway.
    api_key_env: str | None = None
    thinking: bool = False
    temperature: float = 0.0
    seed: int | None = 42
    max_completion_tokens: int = 4096
    timeout_seconds: float = 120.0

    def resolve_profile(self) -> ProviderProfile:
        """Return the effective provider profile with model/base-url overrides."""

        return profile_with_overrides(
            resolve_provider(self.provider),
            model=self.model,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
        )


# Backwards-compatible alias. Older call sites (and pickled configs) reference
# ``QwenConfig``; it is now just the provider-agnostic config with the DashScope
# provider selected by default.
QwenConfig = LLMConfig


@dataclass
class LLMInvocation:
    proposal: LLMMatchProposal | None
    stats: LLMCallStats
    raw_content: str
    parse_error: str | None = None


@dataclass
class AgenticLLMInvocation:
    proposal: AgenticMatchProposal | None
    stats: LLMCallStats
    raw_content: str
    parse_error: str | None = None


MATCH_SYSTEM_PROMPTS = {
    "anchor_v1": """You are a static ONNX graph alignment analyst. Do not execute models and do not invent node IDs. The input is a compact JSON IR for two DAG windows at a mismatch. Immediate parent nodes and parent values are intentionally retained. Find the earliest pair of existing nodes after the mismatch where exact structural matching can safely resume. Prefer the smallest closed mismatch regions. Unknown optimization rules are allowed. Return JSON only with keys: left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation. The region node arrays must exclude the anchor nodes. exit_mapping must be an array of two-string arrays such as [[\"left_value\",\"right_value\"]]. equivalence_claim must be exactly one of exact, algebraic, approximate, unknown. Use null anchors when no defensible rejoin exists.""",
    "proof_v2": """You analyze two static ONNX DAG windows without running inference. Select the earliest rejoin anchor pair whose operators, attributes, arity, shapes, and downstream role are compatible. Respect all parent dependencies. Infer the local N:M optimization between the known mapped entry cut and the proposed anchor, but distinguish a supported ONNX algebraic identity from an opaque custom-op hypothesis. Output strict JSON only with keys: left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation. The region arrays contain only nodes before the anchors and must exclude both anchor nodes. exit_mapping must be an array of [left_value,right_value] pairs, never an object. equivalence_claim must be exactly one of exact, algebraic, approximate, unknown. Every node ID must come from the supplied IR; use null anchors when uncertain.""",
}

MATCH_SYSTEM_PROMPTS["anchor_v1"] = (
    """You are a static ONNX graph alignment analyst. Do not execute models and do not invent node IDs. Find the earliest unchanged node pair after the mismatch, not nodes inside the optimization. When coarse_anchor_pairs is non-empty, you MUST select a pair from that list and prefer the smallest max_hops. Unknown optimization rules are allowed. Return JSON only with keys: left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation. Region arrays contain only nodes before the anchors and exclude the anchors. exit_mapping contains the corresponding anchor INPUT values, not anchor outputs, and must be an array like [["left_value","right_value"]]. Use an exact supported_rule_ids value when applicable. equivalence_claim must be exactly one of exact, algebraic, approximate, unknown. Use null anchors when no defensible rejoin exists."""
)

MATCH_SYSTEM_PROMPTS["proof_v2"] = (
    """You analyze two static ONNX DAG windows without running inference. A rejoin anchor is the first unchanged operation AFTER an N:M optimization region; never select an operation inside the optimization itself. When coarse_anchor_pairs is non-empty, you MUST select a pair from that list and prefer the pair with the smallest max_hops unless its parent dependencies make it invalid. Respect every parent dependency. Infer the local optimization between the mapped entry cut and the anchors, distinguishing supported ONNX algebra from opaque custom-op hypotheses. Output strict JSON only with keys: left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation. Region arrays exclude both anchors. exit_mapping is an array of corresponding anchor INPUT value pairs, never anchor outputs and never an object. rule_id must exactly use a supplied supported_rule_ids value when one applies. equivalence_claim must be exactly one of exact, algebraic, approximate, unknown. Every ID must exist in the supplied IR; use null when uncertain."""
)

MATCH_SYSTEM_PROMPTS["region_v3"] = (
    """You perform static ONNX mismatch-region extraction without model execution. First choose the earliest valid pair from coarse_anchor_pairs. required_parent_nodes are already matched and are present only to explain dependencies: NEVER include them in either region. A region contains every unmatched node from mismatch_frontier up to the selected anchor INPUTS, excluding the anchors. Include a mismatch-frontier node unless that node is itself the selected anchor; this is how an eliminated side may be empty. Do not omit a fused, replacement, or decomposed frontier node. When a rule_catalog pattern matches, list the actual nodes for BOTH pattern sides; a region side may be empty only when that catalog side is explicitly []. Use the exact canonical rule_id from rule_catalog; otherwise use a concise unknown rule ID. exit_mapping must pair the selected anchors' corresponding INPUT values. Return strict JSON only with keys left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation. equivalence_claim is exactly one of exact, algebraic, approximate, unknown. Never invent IDs."""
)

MATCH_SYSTEM_PROMPTS["region_v4"] = (
    """You select a static ONNX rejoin point without model execution. coarse_anchor_pairs contains valid same-signature anchors or virtual __graph_output__:i anchors, sorted by distance, and each pair already contains deterministic closed left_region_nodes and right_region_nodes. Select the earliest pair whose region is plausibly an optimization. Copy that selected pair's region node arrays EXACTLY into the response; do not add, remove, or reinterpret parent nodes. Use rule_catalog to classify the two region operator patterns with the exact canonical rule_id, or use a concise unknown rule ID. required_parent_nodes are already matched context and are never region nodes. For node anchors, exit_mapping pairs corresponding anchor INPUT values; for virtual output anchors, pair the corresponding graph output values. Return strict JSON only with keys left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation. equivalence_claim is exactly one of exact, algebraic, approximate, unknown. Never invent IDs."""
)


AGENTIC_SYSTEM_PROMPTS = {
    "agent_v1": """You independently align two static ONNX DAG windows without model execution. The program has detected the first mismatch but has NOT supplied candidate anchors. Inspect the visible topologically ordered nodes and choose one action: match, expand, or unresolved. Use match only when you can identify the earliest pair of unchanged nodes AFTER the optimization where traversal can safely resume. Use expand when a defensible pair is probably beyond the visible window and agent_control.can_expand is true. Use unresolved only when expansion is unavailable or the visible graphs appear genuinely incompatible. Never invent node or value IDs. Return strict JSON only with keys action, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes.""",
    "agent_v2": """You are the search policy for static ONNX graph rematching. No anchor candidates are supplied: you must discover the rejoin pair from the visible DAGs. Work from mismatch_frontier forward in topological order. A valid rejoin is the earliest visible pair AFTER the changed region with compatible op/domain, attributes, input/output arity, output dtype/shape, and downstream role. Trace backwards from each proposed anchor input to list every unmatched region node; exclude the anchors and required_parent_nodes. All external dynamic region inputs must be explainable by known_value_mapping. Choose action=expand when no complete anchor pair is visible or a likely anchor lies at the window boundary and agent_control.can_expand is true. Choose action=match only after checking both anchors and region closure. Choose action=unresolved only when no defensible match remains. Classify known operator patterns with an exact rule_catalog ID; unknown transformations are allowed but must use equivalence_claim=unknown. Return strict JSON only with keys action, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes. Never invent IDs.""",
    "agent_v3": """Independently find where two static ONNX DAGs rejoin after the current mismatch. The tool gives graph facts and expansion control only; it does not give candidate pairs. Follow this checklist before action=match: (1) scan visible nodes by distance from mismatch_frontier; (2) require the proposed anchors to represent the same unchanged operation, including domain/op, attributes, arity, constant roles, and compatible outputs; (3) verify the anchors are after, not inside, the changed region; (4) trace every dynamic anchor input backwards until required_parent_nodes, graph inputs, or already mapped values, listing all unmatched nodes on both sides; (5) prefer the earliest pair with a closed region and coherent downstream role; (6) use the exact rule_catalog ID only when both operator sequences match. If any needed anchor or dependency is outside the window and agent_control.can_expand is true, return action=expand instead of guessing. action=unresolved is for a sufficiently expanded window with no defensible rejoin. Return one strict JSON object with action, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes. For expand, anchors may be null and expand_hops/expand_nodes must be positive. Never invent IDs or claim proof from similarity alone.""",
    "agent_v4": """Act as the anchor-search policy for two static ONNX DAG windows. No candidate pairs are supplied. First cross-compare visible nodes yourself, starting with the smallest frontier_distance. A rejoin anchor pair is the earliest two visible nodes whose OWN operation descriptors match: domain/op, attributes, input/output arity, constant-input roles, and compatible outputs. Their input VALUE NAMES do not need to be currently mapped, because the unmatched region before the anchors is exactly what will explain those inputs. Example: Identity->Relu versus Relu should select the two Relu nodes as anchors and regions [Identity] versus []; do not skip to graph output. Trace backwards from anchor inputs and list all unmatched nodes before the anchors, excluding required_parent_nodes and the anchors. Use action=expand only when agent_control.can_expand is true AND window_boundary_node_ids or a non-visible graph-output producer shows that graph facts are actually hidden. If the complete visible graph contains a same-operation pair, choose match instead of expanding. Use action=unresolved only when no defensible pair exists in a complete window. Use an exact rule_catalog ID only when the two region op sequences match it; otherwise use an unknown label and equivalence_claim=unknown. Return strict JSON with action, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes. expand_hops and expand_nodes must be integers. Never invent IDs or treat graph output as earlier than a visible unchanged node pair.""",
    "agent_v5": """Find the earliest safe rejoin of two visible static ONNX DAG windows. The tool supplies no candidate pairs. Return concise JSON only; do not include internal debate. HARD RULES for action=match: (1) left_anchor and right_anchor must be the SAME unchanged operation: same domain/op, attributes, arity, constant roles, and compatible outputs. MatMul versus Gemm, Add versus Gemm, or different-shape Relu nodes are NEVER anchor pairs. (2) Every changed node must be BEFORE the anchors. If one side still has a visible transformation node after a proposed anchor, that anchor is inside the optimization and invalid. (3) Known primitive/fused nodes named by rule_catalog are REGION nodes, not anchors. For MatMul->Add versus Gemm, choose the next same-operation pair after Add/Gemm. For Transpose->Relu->Transpose versus Relu, choose a same-operation pair after the outer Transpose, not the inner Relu. (4) If the transformation ends at graph output and there is no later same-operation pair, use matching virtual IDs __graph_output__:i on both sides. (5) Trace backwards from anchor inputs to required_parent_nodes, graph inputs, or known mapped values; list all unmatched nodes, exclude anchors and required parents, and include mismatch_frontier. (6) Never repeat an anchor pair listed in agent_control.rejected_decisions. Search visible pairs in increasing frontier_distance. If no valid pair is visible and hidden graph remains, action=expand. If the complete window has no valid node pair, use virtual output when appropriate, otherwise unresolved. Anchor input value names may differ because the region explains them. Use the exact rule_catalog ID only when region op sequences match; otherwise unknown. Output keys: action, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes. equivalence_claim must be exact, algebraic, approximate, or unknown; expansion fields must be integers. Keep conditions to at most 6 and explanation under 100 words. Never invent IDs or claim proof.""",
    "agent_v6": """Find exactly ONE earliest local rejoin for the current static ONNX mismatch. The tool gives graph facts but no candidate pairs. Return concise JSON only. Search visible left/right nodes by increasing frontier_distance. A valid anchor pair must be the SAME unchanged operation: same domain/op, attributes, arity, constant roles, and compatible outputs. Anchor input value names may differ because the region explains them. All changed nodes for this ONE local transformation must be before the anchors, and mismatch_frontier must be included in the regions. Never combine several later optimizations into one large region and never skip an earlier valid same-operation pair. Example: Identity->Relu followed later by MatMul->Add must first anchor at the first two Relu nodes and region [Identity] versus []; the later linear fusion is handled in a new mismatch loop. MatMul/Gemm, Add/Gemm, or an inner Relu before a trailing Transpose are region nodes, never anchors. For MatMul->Add versus Gemm, use the next same-operation pair after Add/Gemm. For Transpose->Relu->Transpose versus Relu, use a same-operation pair after the outer Transpose. If a terminal transformation has no later unchanged node pair, use matching __graph_output__:i virtual anchors. Trace backwards from anchor inputs to required parents, graph inputs, or known mapped values; list every unmatched node, excluding anchors and required parents. Do not repeat agent_control.rejected_decisions. action=expand only when can_expand=true and hidden graph facts remain; unresolved only for a complete window with no node or virtual-output rejoin. Use exact rule_catalog IDs only for exact region op-sequence matches. Output string anchor IDs, not objects, with keys action, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes. equivalence_claim must be exact, algebraic, approximate, or unknown. Keep conditions <=6 and explanation <100 words. Never invent IDs or claim proof.""",
    "agent_v7": """Independently find ONE earliest local rejoin in two visible static ONNX DAG windows. The tool supplies no candidate pairs. You must perform the candidate scan. Step 1: cross-compare visible left/right nodes and return same_op_pairs_seen as string ID pairs for every plausible pair with the same own domain/op, attributes, arity, constant roles, and compatible outputs, ordered by max frontier_distance. Step 2: reject a seen pair if a changed node remains after it on either path; anchors must be after the whole local transformation. Step 3: choose the earliest remaining pair and trace backwards from its inputs, listing every unmatched node through mismatch_frontier while excluding anchors and required parents. Process only this first local transformation; never merge later optimizations. If any valid same-op pair remains, virtual graph output is forbidden. Use matching __graph_output__:i only when same_op_pairs_seen has no valid post-transformation pair and the transformation ends at output. Examples: Identity->Relu versus Relu selects the two Relu nodes with regions [Identity] versus []; MatMul->Add versus Gemm selects the next same-op nodes after Add/Gemm; Transpose->Relu->Transpose versus Relu must not select the inner Relu because a changed Transpose remains after it. Do not repeat agent_control.rejected_decisions. Expand only when can_expand=true and hidden graph remains. Use exact rule_catalog IDs only when region op sequences match. Return concise strict JSON with action, same_op_pairs_seen, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes. Anchor IDs and pair IDs must be strings. equivalence_claim is exact, algebraic, approximate, or unknown. Never invent IDs or claim proof.""",
    "agent_v8": """Independently find ONE earliest local rejoin in two visible static ONNX DAG windows. The tool supplies no candidate pairs. First cross-compare visible nodes and return same_op_pairs_seen as string ID pairs with the same own domain/op, attributes, arity, constant roles, and compatible outputs, ordered by max frontier_distance. Reject a pair if any changed node remains after it; anchors must be after the whole local transformation. Select the earliest remaining pair and trace backwards from anchor inputs, listing every unmatched node through mismatch_frontier while excluding anchors and required parents. Process only this first transformation and never merge later ones. If a valid same-op pair exists, virtual graph output is forbidden; use matching __graph_output__:i only for a terminal transformation with no valid node pair. Identity->Relu versus Relu anchors at the Relu pair. MatMul->Add versus Gemm anchors at the next same-op pair. Transpose->Relu->Transpose versus Relu cannot anchor at the inner Relu. Rule classification is separate from anchor search: a canonical rule ID requires BOTH region op sequences to exactly match its rule_catalog pattern. [MatMul,Add] versus [FusedLinear] is NOT matmul_add_to_gemm; use unknown_fused_linear when available and equivalence_claim=unknown. Do not infer custom-op semantics from matching weights. Do not repeat rejected_decisions. Expand only when can_expand=true and hidden graph remains. Return concise strict JSON with action, same_op_pairs_seen, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes. IDs must be strings; equivalence_claim is exact, algebraic, approximate, or unknown. Never invent IDs or claim proof.""",
    "agent_v9": """Independently find ONE earliest local structural rejoin in two visible static ONNX DAG windows. The tool supplies no candidate pairs. Cross-compare all visible nodes and return same_op_pairs_seen as string ID pairs with the same own domain/op, attributes, arity, constant roles, and compatible outputs, ordered by max frontier_distance. Anchor search is STRUCTURAL and separate from semantic proof: an unknown custom transformation does not prevent matching a later unchanged pair. For MatMul->Add->Relu versus FusedLinear->Relu, include the two Relu nodes in same_op_pairs_seen, select them as anchors, list [MatMul,Add] versus [FusedLinear] as regions, use rule_id=unknown_fused_linear and equivalence_claim=unknown; the tool will mark it structural rather than proven. Reject a pair only when a changed node remains after it. Select the earliest valid pair, trace backwards from anchor inputs, include every unmatched node through mismatch_frontier, and exclude anchors/required parents. Process only this first transformation; do not merge later ones. If any valid same-op pair exists, virtual output is forbidden. Use matching __graph_output__:i only for a terminal transformation with no valid node pair. Identity->Relu versus Relu anchors at Relu. MatMul->Add versus Gemm anchors after Add/Gemm. Transpose->Relu->Transpose versus Relu cannot anchor at inner Relu. Canonical proven rule IDs require exact BOTH-side rule_catalog patterns; otherwise use an unknown label without refusing the structural anchor. Do not repeat rejected_decisions. Expand only when can_expand=true and hidden graph remains. Return concise strict JSON with action, same_op_pairs_seen, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes. IDs are strings; equivalence_claim is exact, algebraic, approximate, or unknown. Never invent IDs or claim proof for custom ops.""",
    "agent_v10": """Independently find ONE earliest local structural rejoin in two visible static ONNX DAG windows. The tool supplies no candidate pairs. Cross-compare only IDs in visible_anchor_ids and return same_op_pairs_seen as arrays of two string IDs, for example [[\"left_relu\",\"right_relu\"]], for pairs whose own domain/op, attributes, arity, constant roles, and compatible outputs match, ordered by max frontier_distance. The selected non-output anchor pair must appear in same_op_pairs_seen. Nodes in visible_matched_history_ids or required_parent_nodes are context history: never select or report them as anchors. Anchor search is STRUCTURAL and separate from rule naming. Select the earliest valid pair after the whole current transformation, trace backwards from its inputs, include every unmatched node through mismatch_frontier, and exclude anchors and matched history. Process only this first transformation; do not merge later optimizations. Reject an apparent pair when a changed node remains after it. If any valid same-op pair exists, virtual output is forbidden; use matching __graph_output__:i only for a terminal transformation with no valid node pair. Classify the reconstructed region literally from rule_catalog: [Identity] vs [] is identity_elimination; [Transpose,Transpose] vs [] is double_transpose_elimination; [Transpose,Relu,Transpose] vs [Relu] is transpose_elementwise_elimination; [MatMul,Add] vs [Gemm] is matmul_add_to_gemm; [Div,Erf,Add,Mul,Mul] vs [Gelu] is erf_gelu_fusion; [Conv,BatchNormalization] vs [Conv] is conv_batchnorm_fusion. Never label these standard-domain patterns unknown_fused_linear. Reserve unknown_fused_linear for a custom/opaque fused op whose semantics are not in the catalog, and use equivalence_claim=unknown. Expand only when can_expand=true and hidden graph remains. Do not repeat rejected_decisions. Return concise strict JSON with action, same_op_pairs_seen, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes. IDs are strings; equivalence_claim is exact, algebraic, approximate, or unknown. Never invent IDs or claim proof for custom ops.""",
    "agent_v11": """Independently find ONE earliest local structural rejoin in two visible static ONNX DAG windows. The tool supplies no candidate pairs. Cross-compare only IDs in visible_anchor_ids and return same_op_pairs_seen as arrays of two string IDs, for example [[\"left_relu\",\"right_relu\"]], for pairs whose own domain/op, attributes, arity, constant roles, and compatible outputs match, ordered by max frontier_distance. The selected non-output anchor pair must appear in same_op_pairs_seen. Nodes in visible_matched_history_ids or required_parent_nodes are context history: never select or report them as anchors. An anchor is the first unchanged operation AFTER the entire current transformation, never a node inside the transformation and never the last changed node itself. If a visible same-op pair is inside a pattern that still has a changed successor, discard it. Critical example: when the current window shows left Transpose->Relu->Transpose and right Relu, the Relu pair is inside the transformation. If no later same-op pair is visible and window_boundary_node_ids shows hidden successors with can_expand=true, action MUST be expand. Select match only after the complete region, then trace backwards from anchor inputs, include every unmatched node through mismatch_frontier, and exclude anchors and matched history. Process only this first transformation; do not merge later optimizations. If any valid post-transformation same-op pair exists, virtual output is forbidden; use matching __graph_output__:i only for a terminal transformation with no hidden successors and no valid node pair. Classify the region literally from rule_catalog: [Identity] vs [] is identity_elimination; [Transpose,Transpose] vs [] is double_transpose_elimination; [Transpose,Relu,Transpose] vs [Relu] is transpose_elementwise_elimination; [MatMul,Add] vs [Gemm] is matmul_add_to_gemm; [Div,Erf,Add,Mul,Mul] vs [Gelu] is erf_gelu_fusion; [Conv,BatchNormalization] vs [Conv] is conv_batchnorm_fusion. Never label these standard-domain patterns unknown_fused_linear. Reserve unknown_fused_linear for a custom/opaque fused op and use equivalence_claim=unknown. When validator feedback rejects a known-rule region and no different downstream rejoin exists, return unresolved instead of wrapping it in a larger unknown region. Expand only when can_expand=true and hidden graph remains. Do not repeat rejected_decisions. Return concise strict JSON with action, same_op_pairs_seen, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes. IDs are strings; equivalence_claim is exact, algebraic, approximate, or unknown. Never invent IDs or claim proof for custom ops.""",
    "agent_v12": """Independently find ONE earliest local structural rejoin in two visible static ONNX DAG windows. The tool supplies no candidate pairs. Work only on the CURRENT mismatch_frontier. Cross-compare only IDs in visible_anchor_ids and return same_op_pairs_seen as arrays of two string IDs such as [[\"left_relu\",\"right_relu\"]]. The selected non-output anchor pair must appear there. Nodes in visible_matched_history_ids or required_parent_nodes are context history and cannot be anchors. A valid anchor is the first unchanged operation after the CURRENT local transformation. Once that pair is reached, stop: any different operations after the anchor are a NEW mismatch for the next loop and must not invalidate or enlarge the current region. Example: Identity->Relu->later MatMul/Add versus Relu->later Gemm MUST first anchor at the first Relu pair with regions [Identity] versus []; do not merge the later fusion. Distinguish this from Transpose->Relu->Transpose versus Relu: the inner Relu is enclosed by the current transpose_elementwise_elimination pattern, so it is inside the current transformation; if the post-outer-Transpose anchor is hidden and can_expand=true, action MUST be expand. Trace backwards from anchor inputs, include every unmatched node through the current frontier, and exclude anchors and matched history. If a valid post-transformation same-op pair exists, virtual output is forbidden; use matching __graph_output__:i only when the current transformation is terminal, no hidden successors exist, and no valid node pair exists. Classify the current region literally from rule_catalog: [Identity] vs [] identity_elimination; [Transpose,Transpose] vs [] double_transpose_elimination; [Transpose,Relu,Transpose] vs [Relu] transpose_elementwise_elimination; [MatMul,Add] vs [Gemm] matmul_add_to_gemm; [Div,Erf,Add,Mul,Mul] vs [Gelu] erf_gelu_fusion; [Conv,BatchNormalization] vs [Conv] conv_batchnorm_fusion. Never label these standard-domain patterns unknown_fused_linear. Reserve unknown_fused_linear for a custom/opaque fused op and equivalence_claim=unknown. When validator feedback rejects a known-rule region and no different downstream rejoin exists, return unresolved instead of wrapping it in a larger unknown region. Expand only when can_expand=true and facts needed for the CURRENT rejoin are hidden. Do not repeat rejected_decisions. Return concise strict JSON with action, same_op_pairs_seen, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes. Never invent IDs or claim proof for custom ops.""",
    "agent_v13": """Independently find ONE earliest local structural rejoin in two visible static ONNX DAG windows. The tool supplies no candidate pairs. Work only on the CURRENT mismatch_frontier. Cross-compare only IDs in visible_anchor_ids and return same_op_pairs_seen as arrays of two string IDs such as [[\"left_relu\",\"right_relu\"]]. The selected non-output anchor pair must appear there. Nodes in visible_matched_history_ids or required_parent_nodes are context history and cannot be anchors. A valid anchor is the first unchanged operation after the CURRENT local transformation. Once that pair is reached, stop: any different operations after the anchor are a NEW mismatch for the next loop and must not invalidate or enlarge the current region. Example: Identity->Relu->later MatMul/Add versus Relu->later Gemm MUST first anchor at the first Relu pair with regions [Identity] versus []; do not merge the later fusion. Distinguish this from Transpose->Relu->Transpose versus Relu: the inner Relu is enclosed by the current transpose_elementwise_elimination pattern, so it is inside the current transformation; if the post-outer-Transpose anchor is hidden and can_expand=true, action MUST be expand. Trace backwards from anchor inputs, include every unmatched node through the current frontier, and exclude anchors and matched history. If a valid post-transformation same-op pair exists, virtual output is forbidden; use matching __graph_output__:i only when the current transformation is terminal, no hidden successors exist, and no valid node pair exists. Classify the current region literally from rule_catalog: [Identity] vs [] identity_elimination; [Transpose,Transpose] vs [] double_transpose_elimination; [Transpose,Relu,Transpose] vs [Relu] transpose_elementwise_elimination; [MatMul,Add] vs standard ONNX-domain [Gemm] matmul_add_to_gemm; [Div,Erf,Add,Mul,Mul] vs [Gelu] erf_gelu_fusion; [Conv,BatchNormalization] vs [Conv] conv_batchnorm_fusion. A custom-domain op such as com.example::FusedLinear is NOT Gemm even when its name, weights, bias, shapes, or behavior look Gemm-like: label [MatMul,Add] versus that opaque op as unknown_fused_linear with equivalence_claim=unknown. Never infer custom-op semantics from matching constants. When validator feedback rejects a known-rule region and no different downstream rejoin exists, return unresolved instead of wrapping it in a larger unknown region. Expand only when can_expand=true and facts needed for the CURRENT rejoin are hidden. Do not repeat rejected_decisions. Return concise strict JSON with action, same_op_pairs_seen, left_anchor, right_anchor, left_region_nodes, right_region_nodes, exit_mapping, rule_id, equivalence_claim, confidence, conditions, explanation, expand_hops, expand_nodes. Never invent IDs or claim proof for custom ops.""",
}


class LLMClient:
    """OpenAI-compatible chat client that is agnostic to the backend provider.

    The provider (DashScope/Qwen, OpenAI, or a generic OpenAI-compatible
    endpoint) is chosen through :class:`LLMConfig`. Prompt construction, JSON
    extraction, and proposal normalization are provider-independent and shared
    across all backends.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.profile = self.config.resolve_profile()
        self.model = self.profile.default_model
        self.base_url = self.profile.default_base_url
        if not self.model:
            raise RuntimeError(
                f"No model configured for provider {self.profile.name!r}; "
                "pass model=... (or --model) for the openai-compatible provider"
            )
        api_key = self.profile.resolve_api_key(self.config.api_key, self.config.api_key_env)
        if not api_key:
            envs = " or ".join(self.profile.key_env_candidates(self.config.api_key_env))
            raise RuntimeError(
                f"An API key is required for live {self.profile.name!r} calls; set {envs} or pass api_key=..."
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.config.timeout_seconds,
        )

    def propose_match(
        self,
        context: dict[str, Any],
        *,
        context_strategy: str,
        prompt_variant: str = "region_v4",
        feedback: str | None = None,
    ) -> LLMInvocation:
        system_prompt = MATCH_SYSTEM_PROMPTS[prompt_variant]
        user_payload: dict[str, Any] = {
            "instruction": "Analyze this mismatch and output the requested JSON object.",
            "graph_context": context,
        }
        if feedback:
            user_payload["previous_proposal_validation_error"] = feedback
            user_payload["retry_instruction"] = "Correct the proposal without adding graph context or inventing IDs."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
        ]
        started = time.perf_counter()
        content, usage = self._complete(messages, json_mode=not self.config.thinking)
        latency = time.perf_counter() - started
        proposal: LLMMatchProposal | None = None
        parse_error: str | None = None
        try:
            payload = _ProposalPayload.model_validate(self._extract_json(content))
            claim = payload.equivalence_claim
            if claim not in {"exact", "algebraic", "approximate", "unknown"}:
                claim = "unknown"
            proposal = LLMMatchProposal(
                left_anchor=payload.left_anchor,
                right_anchor=payload.right_anchor,
                left_region_nodes=payload.left_region_nodes,
                right_region_nodes=payload.right_region_nodes,
                exit_mapping=self._normalize_exit_mapping(payload.exit_mapping),
                rule_id=payload.rule_id,
                equivalence_claim=claim,  # type: ignore[arg-type]
                confidence=max(0.0, min(1.0, payload.confidence)),
                conditions=self._normalize_conditions(payload.conditions),
                explanation=payload.explanation,
            )
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            parse_error = str(exc)
        stats = LLMCallStats(
            model=self.model,
            context_strategy=context_strategy,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_seconds=latency,
            thinking=self.config.thinking,
            seed=self.config.seed,
            parse_ok=proposal is not None,
            metadata={"prompt_variant": prompt_variant},
        )
        return LLMInvocation(proposal=proposal, stats=stats, raw_content=content, parse_error=parse_error)

    def propose_agentic_match(
        self,
        context: dict[str, Any],
        *,
        context_strategy: str,
        prompt_variant: str = "agent_v9",
        feedback: str | None = None,
    ) -> AgenticLLMInvocation:
        system_prompt = AGENTIC_SYSTEM_PROMPTS[prompt_variant]
        user_payload: dict[str, Any] = {
            "instruction": "Control this mismatch search. Return match, expand, or unresolved as strict JSON.",
            "graph_context": context,
        }
        if feedback:
            user_payload["previous_action_feedback"] = feedback
            user_payload["retry_instruction"] = (
                "Re-evaluate the visible graph facts. Do not invent IDs and request expansion instead of guessing."
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
        ]
        started = time.perf_counter()
        content, usage = self._complete(messages, json_mode=not self.config.thinking)
        latency = time.perf_counter() - started
        proposal: AgenticMatchProposal | None = None
        parse_error: str | None = None
        try:
            payload = _AgenticProposalPayload.model_validate(self._extract_json(content))
            action = payload.action if payload.action in {"match", "expand", "unresolved"} else "unresolved"
            claim = payload.equivalence_claim
            if claim not in {"exact", "algebraic", "approximate", "unknown"}:
                claim = "unknown"
            proposal = AgenticMatchProposal(
                action=action,  # type: ignore[arg-type]
                left_anchor=self._normalize_anchor_id(payload.left_anchor),
                right_anchor=self._normalize_anchor_id(payload.right_anchor),
                left_region_nodes=payload.left_region_nodes,
                right_region_nodes=payload.right_region_nodes,
                exit_mapping=self._normalize_exit_mapping(payload.exit_mapping),
                rule_id=payload.rule_id,
                equivalence_claim=claim,  # type: ignore[arg-type]
                confidence=max(0.0, min(1.0, payload.confidence)),
                conditions=payload.conditions,
                explanation=payload.explanation,
                expand_hops=self._normalize_expand_amount(payload.expand_hops, default=2, maximum=4),
                expand_nodes=self._normalize_expand_amount(payload.expand_nodes, default=8, maximum=16),
                same_op_pairs_seen=self._normalize_id_pairs(payload.same_op_pairs_seen),
            )
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            parse_error = str(exc)
        stats = LLMCallStats(
            model=self.model,
            context_strategy=context_strategy,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_seconds=latency,
            thinking=self.config.thinking,
            seed=self.config.seed,
            parse_ok=proposal is not None,
            metadata={"prompt_variant": prompt_variant, "route": "llm_agent"},
        )
        return AgenticLLMInvocation(
            proposal=proposal,
            stats=stats,
            raw_content=content,
            parse_error=parse_error,
        )

    @staticmethod
    def _normalize_expand_amount(value: Any, *, default: int, maximum: int) -> int:
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return max(1, min(maximum, int(value)))
        if isinstance(value, str):
            try:
                return max(1, min(maximum, int(value)))
            except ValueError:
                return default
        return default

    @staticmethod
    def _normalize_anchor_id(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("id"), str):
            return value["id"]
        return None

    @staticmethod
    def _normalize_conditions(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, dict):
            return [f"{key}={item}" for key, item in value.items()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _normalize_id_pairs(value: Any) -> list[tuple[str, str]]:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                value = [text]
            else:
                value = decoded
        if not isinstance(value, list):
            return []
        if (
            len(value) == 2
            and all(QwenClient._normalize_anchor_id(item) is not None for item in value)
            and not any(isinstance(item, str) and ":" in item for item in value)
        ):
            left = QwenClient._normalize_anchor_id(value[0])
            right = QwenClient._normalize_anchor_id(value[1])
            return [(left, right)] if left is not None and right is not None else []
        pairs: list[tuple[str, str]] = []
        for item in value:
            if isinstance(item, str) and item.startswith("__graph_output__:"):
                continue
            if isinstance(item, str) and ":" in item:
                left_text, right_text = item.split(":", 1)
                left = left_text.strip() or None
                right = right_text.strip() or None
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                left = QwenClient._normalize_anchor_id(item[0])
                right = QwenClient._normalize_anchor_id(item[1])
            elif isinstance(item, dict):
                left = QwenClient._normalize_anchor_id(item.get("left") or item.get("left_anchor"))
                right = QwenClient._normalize_anchor_id(item.get("right") or item.get("right_anchor"))
            else:
                continue
            if left is not None and right is not None:
                pairs.append((left, right))
        return pairs

    @staticmethod
    def _normalize_exit_mapping(value: Any) -> list[tuple[str, str]]:
        if isinstance(value, dict):
            return [(str(left), str(right)) for left, right in value.items()]
        if not isinstance(value, list):
            return []
        result: list[tuple[str, str]] = []
        for pair in value:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                result.append((str(pair[0]), str(pair[1])))
        return result

    def explain(self, summary: dict[str, Any]) -> tuple[str, LLMCallStats]:
        messages = [
            {
                "role": "system",
                "content": "Explain validated ONNX optimization differences in concise English. Separate proven rules, structural-only hypotheses, and rejected/unresolved regions. Do not claim mathematical proof for opaque custom operators.",
            },
            {
                "role": "user",
                "content": json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        started = time.perf_counter()
        content, usage = self._complete(messages, json_mode=False)
        latency = time.perf_counter() - started
        stats = LLMCallStats(
            model=self.model,
            context_strategy="explanation",
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_seconds=latency,
            thinking=self.config.thinking,
            seed=self.config.seed,
            parse_ok=True,
        )
        return content, stats

    def _complete(self, messages: list[dict[str, str]], *, json_mode: bool) -> tuple[str, dict[str, int]]:
        common: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        # ``enable_thinking`` is a DashScope-specific extra-body flag; other
        # OpenAI-compatible servers reject unknown fields, so only send it when
        # the selected provider understands the thinking-stream extension.
        if self.profile.supports_thinking_stream:
            common["extra_body"] = {"enable_thinking": self.config.thinking}
        if self.config.seed is not None:
            common["seed"] = self.config.seed
        if json_mode:
            common["response_format"] = {"type": "json_object"}
            response = self._client.chat.completions.create(**common)
            usage = response.usage.model_dump() if response.usage is not None else {}
            return response.choices[0].message.content or "", usage

        if self.config.thinking and self.profile.supports_thinking_stream:
            stream = self._client.chat.completions.create(
                **common,
                max_completion_tokens=self.config.max_completion_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            parts: list[str] = []
            usage = {}
            for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage.model_dump()
                if chunk.choices:
                    text = chunk.choices[0].delta.content
                    if text:
                        parts.append(text)
            return "".join(parts), usage

        response = self._client.chat.completions.create(**common)
        usage = response.usage.model_dump() if response.usage is not None else {}
        return response.choices[0].message.content or "", usage

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        stripped = content.strip()
        try:
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError("LLM JSON output is not an object")
            return value
        except json.JSONDecodeError:
            pass
        if "```" in stripped:
            for block in stripped.split("```"):
                candidate = block.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                try:
                    value = json.loads(candidate)
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    continue
        start = stripped.find("{")
        if start < 0:
            raise ValueError("No JSON object found in LLM output")
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    value = json.loads(stripped[start : index + 1])
                    if not isinstance(value, dict):
                        raise ValueError("Extracted JSON is not an object")
                    return value
        raise ValueError("Unbalanced JSON object in LLM output")


# Backwards-compatible alias. ``QwenClient`` was the original class name and is
# still imported by existing tests and scripts; it now resolves to the
# provider-agnostic client, which defaults to the DashScope/Qwen backend.
QwenClient = LLMClient
