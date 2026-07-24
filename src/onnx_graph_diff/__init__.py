"""Static ONNX graph-difference discovery."""

from .ir import GraphIR
from .models import ComparisonResult

__all__ = ["ComparisonResult", "GraphIR"]
