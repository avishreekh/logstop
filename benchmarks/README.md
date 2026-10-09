# Benchmarks

This guide covers setup and evaluation with our QMTP and TP2VR benchmarks.

## Install dependencies

Run commands from the repository root using Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r benchmarks/requirements.txt
```

## Query Matching with Temporal Properties (QMTP)

We evaluate temporal queries over objects in videos with our **QMTP-video** benchmark. The videos and frame-level annotations for objects are taken from the [RealTLV dataset](https://github.com/UTAustin-SwarmLab/Temporal-Logic-Video-Dataset). 

This benchmark contains 7471 query-clip samples, with 3,750 satisfied and 3,721 not-satisfied samples. The queries constructed using 15 temporal property templates across five categories and the clips are 10-50 frames long. 

To download videos from the RealTLV dataset and generate length-cropped clips, run:

```bash
python benchmarks/setup_qmtp_video.py --split video
```

The script defaults to `benchmarks/configs/qmtp_video.csv` and writes to `data/qmtp_video/`. 

To change the destination:

```bash
python benchmarks/setup_qmtp_video.py --split video \
  --output-dir /path/to/data/qmtp_video
```

Default config and output paths are resolved relative to the repository. Explicit relative paths are resolved relative to the working directory.

### Run query matching

Install the inference dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

On a server without desktop graphics libraries, if importing OpenCV reports a missing `libGL.so.1`, reinstall the headless build after installing inference dependencies:

```bash
python -m pip install --force-reinstall --no-deps opencv-python-headless==4.12.0.88
```

Evaluate the prepared full benchmark with YOLOv8x:

```bash
python benchmarks/run_qmtp_benchmark.py --split video \
  --videos-dir data/qmtp_video/videos \
  --config benchmarks/configs/qmtp_video.csv \
  --predictor yolov8x.pt \
  --local_threshold 0.5 \
  --smoothing-radii 0 \
  --batch-size 8 \
  --device cuda \
  --output-dir results/qmtp_video
```

Use `--device cpu` when running without a GPU. `--predictor` accepts a supported predictor identifier or checkpoint path.

`--local_threshold` and `--smoothing-radii` also accept JSON mappings such as `--local_threshold '{"car": 0.2, "person": 0.1}'` or `--smoothing-radii '{"car": 1, "person": 2}'`. Omitted properties default to threshold 0.5 and smoothing radius 0.

Results are saved under `--output-dir`:

- `predictions.csv`: original sample fields, LogSTOP score, adaptive threshold, predicted satisfaction, and correctness.
- `summary.json`: balanced accuracy, class recalls, confusion counts, sample/clip counts, config checksum, paths, and matching parameters.
