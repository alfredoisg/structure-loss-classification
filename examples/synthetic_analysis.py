"""Run from repo root after installing the project: python examples/synthetic_analysis.py."""
import argparse
import json

from datasets.synthetic import run_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics, _ = run_analysis(args.output_dir, args.seed)
    print(json.dumps(metrics, indent=2))
    print(f"Outputs: {args.output_dir}/metrics.json and confusion.csv")


if __name__ == "__main__":
    main()
