# LogSTOP: Scores for temporal properties over sequences

This repository contains code for the paper "LogSTOP: Temporal Scores over Prediction Sequences for Matching and Retrieval".

## Project structure

The repository is structured as follows (only the key components are mentioned here):

- `README.md` — project overview and usage
- `requirements.txt` — Python dependencies
- `src/`
  - `logstop.py` — LTL definitions and implementation of LogSTOP
  - `predictors/` — local property predictor implementations (object detectors, etc.)
- `tests/`
  - `test_logstop.py` — unit tests for `logstop` functionality
  - `test_local_property_predictors.py` — tests for predictor implementations

## Query matching with LogSTOP

To evaluate whether a video `video.mp4` matches a temporal property over objects using LogSTOP over predictions from `YOLOv8`, run:

```
python3.10 run_logstop_on_video.py --video_path path/to/video.mp4 \
                                  --fps 5 \
                                  --query "Always (person)" \
                                  --local_property_predictor yolov8x \
                                  --downsampling_smoothing_window 5 \
                                  --batch_size 8 
```

Replace "Always (person)" with any other temporal property over objects that can be detected using YOLO.

Please note that LogSTOP can be used to score temporal properties (queries) over *any* local property set, as long as we have associated local property predictors. Object classes as local properties are only used as an example here.
To evaluate queries over other local properties (actions, concepts, etc.), just define a custom local property predictor (instructions below)!


## Adding a new local property predictor

The local property predictors are defined in `src/predictors`. 

To add a new local property predictor, inherit the `LocalPropertyPredictor` class
from `src/predictors/base.py` and implement the `predict` method.

Please see the `YOLO` class in `src/predictors/object_detectors.py` for an example!