"""Download RealTLV sources and prepare frame-aligned QMTP-video clips."""

import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_REPO = "minkyuchoi/Temporal-Logic-Video-Dataset"
REQUIRED_COLUMNS = {
    "sample_id", "video_id", "source_video_id", "start_frame", "end_frame",
    "query", "satisfied", "temporal_property_template",
    "temporal_property_category", "video_length",
}


def read_config(path):
    """Validate the CSV and collect distinct clips without losing sample rows."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing config columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("The benchmark config is empty")
    samples, clips = set(), {}
    for line, row in enumerate(rows, 2):
        for key in REQUIRED_COLUMNS:
            if not row.get(key):
                raise ValueError(f"CSV line {line}: empty {key}")
        if row["sample_id"] in samples:
            raise ValueError(f"CSV line {line}: duplicate sample_id")
        samples.add(row["sample_id"])
        for key in ("video_id", "source_video_id"):
            value = row[key]
            if value in (".", "..") or "/" in value or "\\" in value:
                raise ValueError(f"CSV line {line}: invalid {key}")
        start, end, length = (int(row[key]) for key in
                              ("start_frame", "end_frame", "video_length"))
        if not 0 <= start < end or length != end - start:
            raise ValueError(f"CSV line {line}: invalid frame range or video_length")
        if row["satisfied"] not in ("true", "false"):
            raise ValueError(f"CSV line {line}: satisfied must be true or false")
        clip = {"source_video_id": row["source_video_id"],
                "start_frame": start, "end_frame": end}
        previous = clips.setdefault(row["video_id"], clip)
        if previous != clip:
            raise ValueError(f"CSV line {line}: conflicting video_id mapping")
    return rows, clips


def index_sources(paths):
    """Prefer the canonical prop1Uprop2 collection when IDs occur twice."""
    index = {}
    for path in sorted(paths, key=lambda p: ("prop1Uprop2" not in Path(p).parts, str(p))):
        index.setdefault(Path(path).stem, path)
    return index


class SourceFiles:
    def __init__(self, source_dir, cache_dir, revision):
        self.cache_dir = Path(cache_dir)
        self.revision = revision
        self.local = (index_sources(Path(source_dir).rglob("*.pkl"))
                      if source_dir else {})
        if source_dir and not Path(source_dir).is_dir():
            raise ValueError(f"Source directory does not exist: {source_dir}")
        self.local_only = source_dir is not None
        self.remote = None

    def get(self, source_id):
        if source_id in self.local:
            return Path(self.local[source_id])
        if self.local_only:
            raise FileNotFoundError(f"Source not found in --source-dir: {source_id}.pkl")
        from huggingface_hub import HfApi, hf_hub_download

        if self.remote is None:
            api = HfApi()
            self.revision = api.dataset_info(DATASET_REPO, revision=self.revision).sha
            paths = api.list_repo_files(DATASET_REPO, repo_type="dataset",
                                        revision=self.revision)
            self.remote = index_sources(
                p for p in paths if p.startswith("tlv_real_dataset/") and p.endswith(".pkl")
            )
        if source_id not in self.remote:
            raise FileNotFoundError(f"Source not found in RealTLV: {source_id}.pkl")
        return Path(hf_hub_download(
            repo_id=DATASET_REPO, repo_type="dataset", revision=self.revision,
            filename=self.remote[source_id], local_dir=str(self.cache_dir),
        ))


def load_source(path):
    # RealTLV publishes Python pickles; only load the official dataset or trusted
    # local copies. The same artifact contains both frames and annotations.
    with Path(path).open("rb") as handle:
        source = pickle.load(handle)
    if not isinstance(source, dict) or not {
        "images_of_frames", "labels_of_frames"
    }.issubset(source):
        raise ValueError(f"Source does not contain RealTLV frames and labels: {path}")
    frames, labels = source["images_of_frames"], source["labels_of_frames"]
    if not len(frames) or len(frames) != len(labels):
        raise ValueError(f"Source frames and labels are not aligned: {path}")
    if any(not isinstance(label, (list, tuple, set)) or
           any(not isinstance(item, str) for item in label) for label in labels):
        raise ValueError(f"Invalid frame-level labels: {path}")
    return frames, labels


def decoded_count(path):
    """Count actual decoded frames, rather than relying on container metadata."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return 0
        count = 0
        while capture.read()[0]:
            count += 1
        return count
    finally:
        capture.release()


def write_video(path, frames, fps, color_order):
    import cv2
    import numpy as np

    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise ValueError("Expected uint8 frames with three color channels")
    height, width = first.shape[:2]
    # mp4v can silently truncate odd dimensions. Reject rather than alter frames.
    if height % 2 or width % 2:
        raise ValueError("MP4 encoding requires even frame dimensions")
    temporary = path.with_name(path.stem + ".partial.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    try:
        if not writer.isOpened():
            raise RuntimeError("Could not open the MP4 video encoder")
        for frame in frames:
            array = np.asarray(frame)
            if array.shape != first.shape or array.dtype != np.uint8:
                raise ValueError("Frame dimensions or dtype vary within the clip")
            if color_order == "rgb":
                array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            writer.write(np.ascontiguousarray(array))
        writer.release()
        if decoded_count(temporary) != len(frames):
            raise RuntimeError(f"Encoded frame count does not match labels: {path.name}")
        temporary.replace(path)
    finally:
        writer.release()
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def setup(config, output_dir, source_dir=None, revision="main", fps=1.0,
          color_order="rgb", overwrite=False):
    rows, clips = read_config(config)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("--fps must be finite and positive")
    if color_order not in ("rgb", "bgr"):
        raise ValueError("--color-order must be rgb or bgr")
    output = Path(output_dir)
    videos = output / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    annotations_dir = output / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    previous_metadata = (json.loads(metadata_path.read_text())
                         if metadata_path.exists() else {})
    resolved_revision = revision
    if (not source_dir and not overwrite and
            previous_metadata.get("requested_revision") == revision and
            previous_metadata.get("dataset_revision")):
        resolved_revision = previous_metadata["dataset_revision"]
    if (previous_metadata and not source_dir and not overwrite and
            previous_metadata.get("requested_revision", revision) != revision):
        raise ValueError("Use --overwrite when changing the dataset revision")
    sources = SourceFiles(source_dir, output / "sources", resolved_revision)
    # Process all crops of a source together so its large pickle loads once.
    grouped = {}
    for video_id, clip in clips.items():
        grouped.setdefault(clip["source_video_id"], []).append((video_id, clip))
    prepared = skipped = 0
    with tqdm(total=len(clips), desc="QMTP-video setup", unit="clip") as progress:
        for source_id, candidates in grouped.items():
            pending = []
            for video_id, clip in candidates:
                target = videos / f"{video_id}.mp4"
                annotation_path = annotations_dir / f"{video_id}.json"
                try:
                    entry = json.loads(annotation_path.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    entry = {}
                if not isinstance(entry, dict):
                    entry = {}
                expected = dict(clip, fps=fps, color_order=color_order,
                                num_frames=clip["end_frame"] - clip["start_frame"])
                valid = (all(entry.get(key) == value for key, value in expected.items())
                         and len(entry.get("labels", [])) == expected["num_frames"]
                         and target.exists() and decoded_count(target) == expected["num_frames"])
                if valid and not overwrite:
                    skipped += 1
                    progress.set_postfix(prepared=prepared, reused=skipped, refresh=False)
                    progress.update(1)
                    continue
                pending.append((video_id, clip, expected, target))
            if not pending:
                continue
            source_path = sources.get(source_id)
            frames, labels = load_source(source_path)
            for video_id, clip, expected, target in pending:
                start, end = clip["start_frame"], clip["end_frame"]
                if end > len(frames):
                    raise ValueError(f"{video_id}: range [{start}, {end}) exceeds {len(frames)} frames")
                write_video(target, frames[start:end], fps, color_order)
                entry = dict(
                    expected, video_id=video_id, video_path=f"videos/{video_id}.mp4",
                    labels=[sorted(set(label)) for label in labels[start:end]],
                )
                write_json(annotations_dir / f"{video_id}.json", entry)
                prepared += 1
                progress.set_postfix(prepared=prepared, reused=skipped, refresh=False)
                progress.update(1)
            del frames, labels
    write_json(metadata_path, {
        "benchmark": "qmtp", "split": "video", "config": Path(config).name,
        "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        "sample_count": len(rows), "video_count": len(clips),
        "dataset_repo": DATASET_REPO,
        "requested_revision": revision,
        "dataset_revision": sources.revision if not source_dir else None,
        "source_dir": str(Path(source_dir).resolve()) if source_dir else None,
        "fps": fps, "fps_role": "playback only; no frame resampling",
        "color_order": color_order, "frame_range_convention": "[start, end)",
    })
    print(f"Complete: {prepared} prepared, {skipped} reused; "
          f"{len(rows)} samples, {len(clips)} videos in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["video"], default="video")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "benchmarks/configs/qmtp_video.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/qmtp_video")
    parser.add_argument("--source-dir", type=Path,
                        help="Use existing RealTLV pickles recursively, without network access")
    parser.add_argument("--revision", default="main", help="Hugging Face revision to download")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Output playback FPS; preserves every sampled frame (default: 1)")
    parser.add_argument("--color-order", choices=["rgb", "bgr"], default="rgb",
                        help="Source frame color order (RealTLV uses RGB)")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild valid existing clips")
    args = parser.parse_args()
    try:
        setup(args.config, args.output_dir, args.source_dir, args.revision,
              args.fps, args.color_order, args.overwrite)
    except (ValueError, OSError, ImportError, RuntimeError, pickle.UnpicklingError) as error:
        parser.exit(1, f"Setup failed: {error}\n")


if __name__ == "__main__":
    main()
