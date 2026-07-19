import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from predict_topk import predict_topk

TOP_K = 5


def get_available_parquet_paths(landmark_dir: Path) -> set:
    """Scans the participant subdirectories to find what files actually exist locally."""
    print(f"Scanning directory for available files: {landmark_dir}")
    # Finds all parquet files inside any subfolder under train_landmark_files
    found_paths = {
        p.relative_to(landmark_dir.parent).as_posix()
        for p in landmark_dir.glob("*/*.parquet")
    }
    print(f"Found {len(found_paths)} local parquet files ready to test.")
    return found_paths


def load_ground_truths(csv_path: Path, available_paths: set, num_samples: int = None, seed: int = 42):
    """Loads matching paths from train.csv, optionally sampling them."""
    df = pd.read_csv(csv_path, usecols=["path", "sequence_id", "sign"])
    
    # Normalize path strings to match OS independent formats
    df["normalized_path"] = df["path"].apply(lambda x: Path(x).as_posix())
    
    # Only keep rows where the parquet file actually exists in your local folder structure
    df_filtered = df[df["normalized_path"].isin(available_paths)].reset_index(drop=True)
    
    if len(df_filtered) == 0:
        raise FileNotFoundError("No matching local parquet files were found in train.csv grid.")

    if num_samples and num_samples < len(df_filtered):
        print(f"Sampling {num_samples} out of {len(df_filtered)} available records...")
        df_filtered = df_filtered.sample(n=num_samples, random_state=seed).reset_index(drop=True)
    else:
        print(f"Processing ALL {len(df_filtered)} available records...")

    return df_filtered.to_dict(orient="records")


def evaluate_predictions(rows, dataset_root: Path, top_k: int = TOP_K):
    results = []
    confusion_counter = Counter()

    for row in rows:
        parquet_path = dataset_root / row["path"]
        sequence_id = row["sequence_id"]
        true_sign = row["sign"]

        # Run model inference
        prediction_payload = predict_topk(parquet_path, k=top_k)
        predictions = prediction_payload["output"]["predictions"]

        top_labels = [prediction["label"] for prediction in predictions]
        top_1 = top_labels[0]
        top_5_contains = true_sign in top_labels

        if top_1 != true_sign:
            confusion_counter[(true_sign, top_1)] += 1

        result_entry = {
            "sequence_id": sequence_id,
            "true_sign": true_sign,
            "top_1_prediction": top_1,
            "is_top_1_correct": top_1 == true_sign,
            "is_in_top_5": top_5_contains,
        }

        for pred in predictions:
            rank = pred["rank"]
            result_entry[f"rank_{rank}_label"] = pred["label"]
            result_entry[f"rank_{rank}_confidence"] = pred["confidence"]

        results.append(result_entry)
        print_file_result(sequence_id, true_sign, predictions, top_5_contains)

    return results, confusion_counter


def print_file_result(sequence_id, true_sign, predictions, top_5_contains):
    print("=" * 80)
    print(f"Sequence ID: {sequence_id}")
    print(f"Ground truth: {true_sign}")
    print("Top 5 predictions:")

    for prediction in predictions:
        rank = prediction["rank"]
        label = prediction["label"]
        confidence = prediction["confidence"]
        print(f"  {rank}. {label:<20} confidence={confidence:.4f}")

    caught_text = "YES" if top_5_contains else "NO"
    print(f"Caught in top {len(predictions)}: {caught_text}")


def summarize_results(results, confusion_counter):
    total = len(results)
    if total == 0:
        print("No predictions were evaluated.")
        return 0.0, 0.0

    top_1_count = sum(1 for item in results if item["is_top_1_correct"])
    top_5_count = sum(1 for item in results if item["is_in_top_5"])

    top_1_accuracy = top_1_count / total * 100
    top_5_accuracy = top_5_count / total * 100

    print("\n" + "#" * 80)
    print("Evaluation summary")
    print("#" * 80)
    print(f"Total files evaluated: {total}")
    print(f"Top-1 Accuracy: {top_1_accuracy:.2f}%")
    print(f"Top-{TOP_K} Accuracy: {top_5_accuracy:.2f}%")

    if confusion_counter:
        print("\nTop Confused Signs (true -> predicted):")
        for (true_sign, predicted_sign), count in confusion_counter.most_common(10):
            print(f"  {true_sign} -> {predicted_sign}: {count}")
            
    return top_1_accuracy, top_5_accuracy


def save_results_to_csv(results, top_1_acc, output_dir: Path):
    if not results:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_{timestamp}_acc{top_1_acc:.1f}.csv"
    output_path = output_dir / filename

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Matrix saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SignBridge model accuracy matrix across parquet directory.")
    parser.add_argument(
        "--samples", 
        type=int, 
        default=None, 
        help="Number of random samples to test. Omit this flag to automatically run all files found in directory."
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_root = script_dir.parent / "dataset"
    landmark_dir = dataset_root / "train_landmark_files"
    csv_path = dataset_root / "train.csv"

    # Step 1: Dynamic discovery based on folder architecture structure map
    available_local_files = get_available_parquet_paths(landmark_dir)

    # Step 2: Load mapping from train.csv filtered down dynamically
    rows = load_ground_truths(csv_path, available_local_files, num_samples=args.samples)

    # Step 3: Compute and matrix mapping pipeline
    print(f"Evaluating {len(rows)} parquet data logs...")
    results, confusion_counter = evaluate_predictions(rows, dataset_root, top_k=TOP_K)

    top_1_acc, top_5_acc = summarize_results(results, confusion_counter)
    save_results_to_csv(results, top_1_acc, script_dir)


if __name__ == "__main__":
    main()