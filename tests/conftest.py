from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_graph_diff.fixtures import generate_fixture_corpus


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    generate_fixture_corpus(tmp_path)
    return tmp_path


@pytest.fixture
def corpus(corpus_dir: Path) -> list[dict]:
    return json.loads((corpus_dir / "corpus.json").read_text(encoding="utf-8"))
