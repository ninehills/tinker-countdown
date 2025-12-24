import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path):
    steps_train, train_correct = [], []
    steps_test, test_correct = [], []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = record.get("step")
            if step is None or step > 200:
                continue
            tr = record.get("env/all/correct")
            te = record.get("test/env/all/correct")
            if tr is not None:
                steps_train.append(step)
                train_correct.append(tr)
            if te is not None:
                steps_test.append(step)
                test_correct.append(te)
    return steps_train, train_correct, steps_test, test_correct


def plot_metrics(metrics_path: Path, output: Path):
    steps_train, train_correct, steps_test, test_correct = load_metrics(metrics_path)
    if not steps_train and not steps_test:
        raise ValueError(f"No valid records in {metrics_path}")

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        steps_test,
        test_correct,
        color="red",
        label="test",
        linestyle="-",
        marker="o",
        markersize=4,
        linewidth=1.8,
    )
    plt.plot(
        steps_train,
        train_correct,
        color="green",
        alpha=0.4,
        label="train",
        linestyle="--",
        marker="o",
        markersize=2.5,
        linewidth=1.0,
    )

    # Simple exponential moving average for train (eta smoothing)
    if train_correct:
        eta = 0.2
        smoothed = []
        last = train_correct[0]
        for v in train_correct:
            last = eta * v + (1 - eta) * last
            smoothed.append(last)
        plt.plot(steps_train, smoothed, color="green", linewidth=2.3, alpha=0.9)

    plt.axhline(0.420, color="red", linestyle="--", linewidth=1, label="Qwen3-4B-Instruct-2507")
    plt.axhline(0.575, color="purple", linestyle="--", linewidth=1, label="Qwen3-235B-A22B-Instruct-2507")

    plt.xlim(0, 200)
    plt.ylim(0.4, 0.9)
    plt.xlabel("step")
    plt.ylabel("reward_correct")
    plt.title("Countdown RL Correct Reward vs Step")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved plot to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot reward_correct over steps from metrics.jsonl")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("metrics.jsonl"),
        help="Path to metrics.jsonl file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reward_correct.png"),
        help="Path to save the plot image",
    )
    args = parser.parse_args()
    plot_metrics(args.metrics, args.output)


if __name__ == "__main__":
    main()
