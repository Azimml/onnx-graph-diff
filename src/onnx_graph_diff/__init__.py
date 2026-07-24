"""Static ONNX graph-difference discovery.

Public entry points:

* :class:`GraphIR` -- load an ONNX model into the normalized graph IR.
* :class:`FingerprintMatcher` -- the deterministic, no-API-key comparison route.
* :class:`StaticEquivalenceValidator` -- the independent opset-aware checker.
* :class:`ComparisonResult` -- the JSON-serializable comparison output.
* :func:`build_workflow` -- assemble the load -> match -> explain state machine.
* :func:`main` -- the ``onnx-graph-diff`` command-line entry point.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .cli import main
from .equivalence import StaticEquivalenceValidator
from .ir import GraphIR
from .matcher import FingerprintMatcher, MatcherConfig
from .models import ComparisonResult
from .workflow import build_workflow

try:
    __version__ = version("onnx-graph-diff")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = [
    "ComparisonResult",
    "FingerprintMatcher",
    "GraphIR",
    "MatcherConfig",
    "StaticEquivalenceValidator",
    "__version__",
    "build_workflow",
    "main",
]
