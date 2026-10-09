"""Run LogSTOP query matching on prepared QMTP-video clips."""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from benchmarks.setup_qmtp_video import read_config, write_json
from src.evaluation import match_query, validate_thresholds
from src.predictors.base import get_local_property_predictor
from src.utils.ltl import extract_local_properties, parse_formula_from_string


def parse_thresholds(value):
    """Accept a scalar or JSON mapping of class-specific thresholds."""
    try:
        thresholds = json.loads(value)
        validate_thresholds(thresholds)
        return thresholds
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_smoothing_radii(value):
    """Accept an integer radius or a JSON object of per-property radii."""
    try:
        radii = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("Use an integer or JSON object for smoothing radii") from error
    values = radii.values() if isinstance(radii, dict) else [radii]
    if any(type(radius) is not int or radius < 0 for radius in values):
        raise argparse.ArgumentTypeError("Smoothing radii must be non-negative integers")
    return radii


def classification_metrics(results):
    counts = dict(tp=0, tn=0, fp=0, fn=0)
    for result in results:
        true = result["satisfied"] == "true"
        predicted = result["predicted_satisfied"] == "true"
        key = ("tp" if predicted else "fn") if true else ("fp" if predicted else "tn")
        counts[key] += 1
    positives, negatives = counts["tp"] + counts["fn"], counts["tn"] + counts["fp"]
    positive_recall = counts["tp"] / positives if positives else None
    negative_recall = counts["tn"] / negatives if negatives else None
    recalls = [value for value in (positive_recall, negative_recall) if value is not None]
    return dict(counts, sample_count=len(results), satisfied_count=positives,
                not_satisfied_count=negatives, satisfied_recall=positive_recall,
                not_satisfied_recall=negative_recall,
                balanced_accuracy=sum(recalls) / len(recalls))


def run_benchmark(videos_dir, config, predictor="yolov8x.pt", local_threshold=0.5,
                  smoothing_radii=0, batch_size=8, device="cpu", output_dir=None,
                  image_size="frame"):
    if str(image_size) not in ("640", "frame"):
        raise ValueError("image_size must be 640 or frame")
    validate_thresholds(local_threshold)
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    # Validate smoothing even if the caller bypasses argparse.
    parse_smoothing_radii(json.dumps(smoothing_radii))
    rows, clips = read_config(config)
    videos_dir = Path(videos_dir)
    formulas, grouped, properties = {}, {}, {}
    for row in rows:
        video_id = row["video_id"]
        path = videos_dir / f"{video_id}.mp4"
        if not path.is_file():
            raise FileNotFoundError(f"Missing video: {path}")
        formula = parse_formula_from_string(row["query"])
        props = extract_local_properties(formula)
        if not props:
            raise ValueError(f"Sample {row['sample_id']} has no local properties")
        formulas[row["sample_id"]] = formula
        grouped.setdefault(video_id, []).append(row)
        properties.setdefault(video_id, set()).update(props)
    all_properties = sorted(set().union(*properties.values()))
    effective_thresholds = {
        prop: local_threshold.get(prop, 0.5) if isinstance(local_threshold, dict)
        else local_threshold for prop in all_properties
    }
    local_predictor = get_local_property_predictor(predictor)
    local_predictor.image_size = "frame" if image_size == "frame" else 640
    predictions = {}
    with tqdm(total=len(rows), desc="QMTP-video", unit="sample") as progress:
        for video_id, samples in grouped.items():
            trace = local_predictor.generate_trace(
                str(videos_dir / f"{video_id}.mp4"),
                local_properties=sorted(properties[video_id]),
                batch_size=batch_size, device=device,
            )
            expected_length = clips[video_id]["end_frame"] - clips[video_id]["start_frame"]
            for prop in properties[video_id]:
                if prop not in trace or len(trace[prop]) != expected_length:
                    raise ValueError(f"{video_id}: prediction trace for {prop!r} "
                                     f"must contain {expected_length} frames")
            for row in samples:
                result = match_query(trace, formulas[row["sample_id"]],
                                     local_threshold, smoothing_radii)
                predicted = str(result["matched"]).lower()
                predictions[row["sample_id"]] = dict(
                    row, score=result["score"], adaptive_threshold=result["threshold"],
                    predicted_satisfied=predicted,
                    correct=str(predicted == row["satisfied"]).lower(),
                )
                progress.update(1)
    results = [predictions[row["sample_id"]] for row in rows]
    metrics = classification_metrics(results)
    summary = dict(
        metrics, benchmark="qmtp", split="video", video_count=len(clips),
        config=str(Path(config).resolve()),
        config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        videos_dir=str(videos_dir.resolve()), predictor=predictor,
        local_threshold=local_threshold, smoothing_radii=smoothing_radii,
        threshold_by_class_effective=effective_thresholds,
        batch_size=batch_size, device=device, image_size=image_size,
    )
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(results)
        write_json(output / "summary.json", summary)
    print(f"Evaluated {len(rows)} samples across {len(clips)} videos")
    print(f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f} "
          f"({metrics['balanced_accuracy']:.2%})")
    return results, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["video"], default="video")
    parser.add_argument("--videos-dir", "--videos_dir", type=Path,
                        default=REPO_ROOT / "data/qmtp_video/videos")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "benchmarks/configs/qmtp_video.csv")
    parser.add_argument("--predictor", "--local_property_predictor", default="yolov8x.pt")
    parser.add_argument("--local_threshold", type=parse_thresholds, default=0.5,
                        help="Local threshold or JSON mapping of class thresholds (default: 0.5)")
    parser.add_argument("--smoothing-radii", "--smoothing_radii", "--smoothing-radius",
                        "--smoothing_radius", type=parse_smoothing_radii, default=0,
                        help='Non-negative radius or JSON mapping, e.g. \'{"car": 1}\'')
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", choices=["640", "frame"], default="frame",
                        help="YOLO inference size: 640 or each video's frame dimensions (default: frame)")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/qmtp_video")
    args = parser.parse_args()
    try:
        run_benchmark(args.videos_dir, args.config, args.predictor, args.local_threshold,
                      args.smoothing_radii, args.batch_size, args.device, args.output_dir,
                      args.image_size)
    except (ValueError, OSError, ImportError, argparse.ArgumentTypeError) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")


if __name__ == "__main__":
    main()
