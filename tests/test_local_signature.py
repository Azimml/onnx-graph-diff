from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx_graph_diff.ir import GraphIR


def _single_node_graph(node: onnx.NodeProto, *, initializers=None) -> GraphIR:
    inp = helper.make_tensor_value_info("in", TensorProto.FLOAT, [2, 2])
    out = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, [2, 2])
    graph = helper.make_graph([node], "g", [inp], [out], initializer=initializers or [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    onnx.checker.check_model(model)
    return GraphIR.from_model(model)


def test_local_signature_ignores_value_and_node_names():
    left = _single_node_graph(helper.make_node("Relu", ["in"], ["a"], name="relu_left"))
    right = _single_node_graph(helper.make_node("Relu", ["in"], ["b"], name="relu_right"))
    assert left.local_signature("relu_left") == right.local_signature("relu_right")


def test_local_signature_depends_on_attributes():
    first = _single_node_graph(helper.make_node("Transpose", ["in"], ["out"], name="t", perm=[1, 0]))
    second = _single_node_graph(helper.make_node("Transpose", ["in"], ["out"], name="t", perm=[0, 1]))
    assert first.local_signature("t") != second.local_signature("t")


def test_local_signature_depends_on_constant_inputs():
    def build(scale: float) -> GraphIR:
        const = numpy_helper.from_array(np.full((2, 2), scale, dtype=np.float32), name="c")
        node = helper.make_node("Add", ["in", "c"], ["out"], name="add")
        return _single_node_graph(node, initializers=[const])

    one, two = build(1.0), build(2.0)
    assert one.local_signature("add") != two.local_signature("add")
    # ...but the digest can be excluded when only the structural shape matters.
    assert one.local_signature("add", include_constants=False) == two.local_signature("add", include_constants=False)
