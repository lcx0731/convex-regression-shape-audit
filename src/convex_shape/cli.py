"""Fit weighted convex regression from a small JSON file."""

import argparse
import json
from pathlib import Path
import sys

from .core import fit, predict


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        with args.input.open("rb") as handle:
            data = handle.read(1024*1024+1)
        if len(data) > 1024*1024:
            raise ValueError("input exceeds 1 MiB")
        inputs = json.loads(data, object_pairs_hook=_pairs)
        if not isinstance(inputs, dict) or not {"x", "y"} <= inputs.keys() or inputs.keys()-{"x", "y", "weights", "predict_x", "max_steps"}:
            raise ValueError("missing or unknown input fields")
        if "weights" in inputs and inputs["weights"] is None:
            raise ValueError("omit weights for unit weights; null is not a vector")
        result = fit(inputs["x"], inputs["y"], inputs.get("weights"), max_steps=inputs.get("max_steps", 10000))
        if "predict_x" in inputs and result["status"] == "converged":
            result["predictions"] = predict(result, inputs["predict_x"]).tolist()
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] == "converged" else 3
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        print(f"convex-fit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
