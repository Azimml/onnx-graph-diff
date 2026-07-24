from __future__ import annotations

import argparse
import json

from onnx_graph_diff.model_zoo import generate_model_zoo


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate seeded ONNX architectures and rewrite combinations for tests."
    )
    parser.add_argument("--output", default="artifacts/model-zoo")
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    corpus = generate_model_zoo(args.output, seed=args.seed)
    print(json.dumps({"output": args.output, "variants": len(corpus)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
