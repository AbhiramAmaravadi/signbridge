from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter

import numpy as np

from predict_topk import predict_topk


@dataclass(frozen=True)
class BenchmarkSample:
    path: Path
    sequence_id: str
    ground_truth: str


@dataclass(frozen=True)
class BenchmarkResult:
    sequence_id: str
    ground_truth: str
    latency_ms: float
    top1_prediction: str
    correct: bool | None


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Benchmark SignBridge parquet-to-TFLite inference latency."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "dataset",
        help="Dataset root containing train.csv and train_landmark_files/.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Optional maximum number of parquet files to evaluate.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup predictions to run before recording timings.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples before applying --samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_results",
        help="Directory where result and summary CSV files are written.",
    )
    return parser.parse_args()


def load_samples(dataset_root: Path) -> list[BenchmarkSample]:
    train_csv = dataset_root / "train.csv"
    landmark_root = dataset_root / "train_landmark_files"

    if train_csv.exists():
        samples: list[BenchmarkSample] = []
        with train_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                path = dataset_root / row["path"]
                if path.exists():
                    samples.append(
                        BenchmarkSample(
                            path=path,
                            sequence_id=str(row.get("sequence_id") or path.stem),
                            ground_truth=str(row.get("sign") or ""),
                        )
                    )
        return samples

    return [
        BenchmarkSample(path=path, sequence_id=path.stem, ground_truth="")
        for path in sorted(landmark_root.glob("*/*.parquet"))
    ]


def select_samples(
    samples: list[BenchmarkSample],
    sample_count: int | None,
    shuffle: bool,
    seed: int,
) -> list[BenchmarkSample]:
    selected = list(samples)
    if shuffle:
        random.Random(seed).shuffle(selected)

    if sample_count is not None:
        if sample_count <= 0:
            raise ValueError("--samples must be greater than zero when provided.")
        selected = selected[:sample_count]

    return selected


def run_warmup(samples: list[BenchmarkSample], warmup_count: int, top_k: int) -> None:
    if warmup_count <= 0 or not samples:
        return

    for sample in samples[:warmup_count]:
        predict_topk(sample.path, k=top_k)


def benchmark_sample(sample: BenchmarkSample, top_k: int) -> BenchmarkResult:
    started_at = perf_counter()
    payload = predict_topk(sample.path, k=top_k)
    latency_ms = (perf_counter() - started_at) * 1000.0

    predictions = payload["output"]["predictions"]
    top1 = predictions[0]["label"] if predictions else ""
    correct = None if not sample.ground_truth else top1 == sample.ground_truth

    return BenchmarkResult(
        sequence_id=sample.sequence_id,
        ground_truth=sample.ground_truth,
        latency_ms=latency_ms,
        top1_prediction=top1,
        correct=correct,
    )


def summarize(results: list[BenchmarkResult], total_runtime_sec: float) -> dict[str, float | int]:
    if not results:
        raise ValueError("No benchmark results were collected.")

    latencies = [result.latency_ms for result in results]
    return {
        "files_evaluated": len(results),
        "average_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "std_deviation_ms": stdev(latencies) if len(latencies) > 1 else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "total_runtime_sec": total_runtime_sec,
        "samples_per_second": len(results) / total_runtime_sec if total_runtime_sec > 0 else 0.0,
        "accuracy": (
            sum(1 for result in results if result.correct is True)
            / sum(1 for result in results if result.correct is not None)
            if any(result.correct is not None for result in results)
            else 0.0
        ),
    }


def write_result_csv(path: Path, results: list[BenchmarkResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence_id", "ground_truth", "latency_ms", "top1_prediction", "correct"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "sequence_id": result.sequence_id,
                    "ground_truth": result.ground_truth,
                    "latency_ms": f"{result.latency_ms:.4f}",
                    "top1_prediction": result.top1_prediction,
                    "correct": "" if result.correct is None else result.correct,
                }
            )


def write_summary_csv(path: Path, summary: dict[str, float | int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in summary.items():
            writer.writerow({"metric": metric, "value": value})


def print_summary(summary: dict[str, float | int], result_csv: Path, summary_csv: Path) -> None:
    print("==============================")
    print("Performance Summary")
    print("==============================")
    print()
    print(f"Files evaluated : {summary['files_evaluated']}")
    print()
    print(f"Average latency : {summary['average_latency_ms']:.2f} ms")
    print(f"Median latency  : {summary['median_latency_ms']:.2f} ms")
    print(f"Min latency     : {summary['min_latency_ms']:.2f} ms")
    print(f"Max latency     : {summary['max_latency_ms']:.2f} ms")
    print(f"95th percentile : {summary['p95_latency_ms']:.2f} ms")
    print(f"Std deviation   : {summary['std_deviation_ms']:.2f} ms")
    print()
    print(f"Throughput      : {summary['samples_per_second']:.2f} samples/sec")
    print(f"Accuracy        : {summary['accuracy']:.4f}")
    print()
    print(f"Results CSV     : {result_csv}")
    print(f"Summary CSV     : {summary_csv}")


def main() -> None:
    args = parse_args()
    samples = select_samples(
        load_samples(args.dataset_root),
        sample_count=args.samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    if not samples:
        raise SystemExit(f"No parquet files found under {args.dataset_root}.")

    run_warmup(samples, warmup_count=args.warmup, top_k=args.top_k)

    started_at = perf_counter()
    results = [benchmark_sample(sample, top_k=args.top_k) for sample in samples]
    total_runtime_sec = perf_counter() - started_at
    summary = summarize(results, total_runtime_sec)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.output_dir / f"benchmark_results_{timestamp}.csv"
    summary_csv = args.output_dir / f"benchmark_summary_{timestamp}.csv"

    write_result_csv(result_csv, results)
    write_summary_csv(summary_csv, summary)
    print_summary(summary, result_csv, summary_csv)


if __name__ == "__main__":
    main()
