"""
Unit tests for local property predictors.

To run the tests:
python -m unittest tests.test_local_property_predictors.TestLocalPropertyPredictors
"""

import unittest
from unittest.mock import patch
import numpy as np
from src.predictors.object_detectors import YOLO
import torch

class TestLocalPropertyPredictors(unittest.TestCase):
    @patch("ultralytics.YOLO.predict")
    @patch("ultralytics.YOLO.__init__")
    @patch("ultralytics.YOLO.names", new_callable=lambda: {0: "person", 1: "car", 2: "bicycle"})
    def test_yolov8_predict(self, mock_names, mock_init, mock_predict):
        mock_init.return_value = None
        mock_predict.return_value = [
            [   type('Detections', (), {
                    'boxes': type('Boxes', (), {
                        'conf': torch.tensor([0.95, 0.80, 0.60]),
                        'cls': torch.tensor([0, 1, 2])
                    })()
                })()
            ],
        ]
        
        # Initialize YOLO predictor
        predictor = YOLO(model_path="yolov11x.pt")
        
        # Dummy frames (e.g., 2 frames of 640x480 with 3 channels)
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames = [frame1, frame2]
        
        # Propositions (object classes to detect)
        local_properties = ["person", "car", "bicycle"]
        
        # Perform prediction
        results = predictor.predict(frames, local_properties, batch_size=1)
        
        # Check results format
        self.assertEqual(len(results), 2)  # Should have results for each frame
        for result in results:
            self.assertIsInstance(result, dict)
            for prop in local_properties:
                self.assertIn(prop, result.keys())  # prop may or may not be detected

    @patch("ultralytics.YOLO.predict")
    @patch("ultralytics.YOLO.__init__")
    @patch("ultralytics.YOLO.names", new_callable=lambda: {0: "person", 1: "car", 2: "bicycle"})
    def test_generate_trace(self, mock_names, mock_init, mock_predict):
        mock_init.return_value = None
        mock_predict.return_value = [
            [   type('Detections', (), {
                    'boxes': type('Boxes', (), {
                        'conf': torch.tensor([0.95, 0.80]),
                        'cls': torch.tensor([0, 1])
                    })()
                })()
            ],
        ]
        # Initialize YOLO predictor
        predictor = YOLO(model_path="yolov11x.pt")
        
        # Dummy frames (e.g., 2 frames of 640x480 with 3 channels)
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames = [frame1, frame2]
        
        # Propositions (object classes to detect)
        local_properties = ["person", "car", "bicycle"]
        
        # Generate trace
        trace = predictor.generate_trace(frames, local_properties, batch_size=1)
        
        # Check trace format
        self.assertIsInstance(trace, dict)
        for prop in local_properties:
            self.assertIn(prop, trace.keys())
            self.assertEqual(len(trace[prop]), 2)  # Should have scores for each frame