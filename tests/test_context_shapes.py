from __future__ import annotations

from onnx_graph_diff.context import GraphContextBuilder

_shape_compatible = GraphContextBuilder._shape_compatible


def test_unknown_shape_is_treated_as_compatible():
    # A missing shape (None) must never block an anchor match.
    assert _shape_compatible(None, (1, 2)) is True
    assert _shape_compatible((1, 2), None) is True
    assert _shape_compatible(None, None) is True


def test_matching_static_shapes_are_compatible():
    assert _shape_compatible((2, 3), (2, 3)) is True


def test_different_rank_is_incompatible():
    assert _shape_compatible((2, 3), (2, 3, 1)) is False


def test_conflicting_static_dimension_is_incompatible():
    assert _shape_compatible((2, 3), (2, 4)) is False


def test_symbolic_dimensions_do_not_conflict():
    # Dynamic ("batch") dims are compatible with any peer, static or symbolic.
    assert _shape_compatible(("batch", 3), (7, 3)) is True
    assert _shape_compatible(("batch", 3), ("N", 3)) is True
