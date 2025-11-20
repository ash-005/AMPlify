"""Quick tuning runner — runs multiple experiments and tracks best ROC AUC."""
import argparse
import subprocess
import json
from pathlib import Path


def run_experiment(config_path: str):
    """Run a single experiment and extract ROC AUC from the output."""
    print(f"\n{'='*60}")
    print(f"Running: {config_path}")
    print('='*60)
    result = subprocess.run(
        ["python", "-m", "amplify.experiments.amplify_v2.train", "--config", config_path],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"Failed: {config_path}")
        return None

    # Try to extract ROC AUC from logs
    log_dir = Path("amplify/experiments/logs")
    latest_metric = sorted(log_dir.glob("run_*_metrics.json"))[-1] if list(log_dir.glob("run_*_metrics.json")) else None
    if latest_metric:
        with open(latest_metric) as f:
            metrics = json.load(f)
            roc_auc = metrics.get("roc_auc", 0.0)
            print(f"Result: ROC AUC = {roc_auc:.4f}")
            return roc_auc
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=[
        "amplify/experiments/amplify_v2/config_tune_v1.yaml",
        "amplify/experiments/amplify_v2/config_tune_v2.yaml",
        "amplify/experiments/amplify_v2/config_tune_v3.yaml",
    ])
    parser.add_argument("--target", type=float, default=0.85)
    args = parser.parse_args()

    results = {}
    best_roc = 0.0

    for config_path in args.configs:
        roc = run_experiment(config_path)
        if roc is not None:
            results[config_path] = roc
            best_roc = max(best_roc, roc)
            print(f"\nCurrent best ROC AUC: {best_roc:.4f} (target: {args.target:.2f})")

            if best_roc >= args.target:
                print(f"\n✓ Reached target ROC AUC {args.target:.2f}!")
                break

    print(f"\n{'='*60}")
    print("Summary:")
    for config, roc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        target_mark = " ✓" if roc >= args.target else ""
        print(f"  {config}: ROC AUC = {roc:.4f}{target_mark}")
    print(f"Best: {best_roc:.4f}")
    print('='*60)


if __name__ == "__main__":
    main()
