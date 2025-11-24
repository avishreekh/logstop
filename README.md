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

## Adding a new local property predictor

The local property predictors are defined in `src/predictors`. 
In order to add a new local property predictor, inherit the `LocalPropertyPredictor` class
from `src/predictors/base.py` and implement the `predict` method.
Please see the `YOLO` class in `src/predictors/object_detectors.py` for an example!