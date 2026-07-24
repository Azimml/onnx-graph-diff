from __future__ import annotations

import argparse
import json

from onnx_graph_diff.fixtures import generate_fixture_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate controlled ONNX graph-diff test fixtures.")
    parser.add_argument("--output", default="artifacts/fixtures")
    args = parser.parse_args()
    corpus = generate_fixture_corpus(args.output)
    print(json.dumps({"output": args.output, "models": len(corpus)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
