from typing import List, Dict
import numpy as np
from .base import LocalPropertyPredictor
    
class YOLO(LocalPropertyPredictor):
    def __init__(self, model_path: str = "yolov8x.pt"):
        super().__init__(model_path)
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        print(f"Classes: {self.model.names}")

    def predict(self, frames: List[np.ndarray], local_properties: List[str], batch_size: int = 8) -> List[Dict[str, float]]:
        """
        Detect objects in a batch of frames using YOLOv8.
        Args:
            frames (List[np.ndarray]): List of frames as numpy arrays.
            local_properties (List[str]): List of local properties (objects) to detect.
            batch_size (int): Batch size for processing frames.
        Returns:
            List[Dict[str, float]]: A list where each element corresponds to a frame and contains a dictionary mapping detected object classes to their confidence scores.
                                    An example of the list for 2 frames could be: [{'person': 0.95, 'car': 0.80}, {'bicycle': 0.90}]
        """
        results = []
        for i in range(0, len(frames), batch_size):
            frames_batch = frames[i:i+batch_size]
            # YOLOv8 can process batches directly
            detections = self.model.predict(frames_batch, conf=0, imgsz=(frames_batch[0].shape[0], frames_batch[0].shape[1]))
            for detection in detections:
                detected_objects_with_scores = {}
                for det in detection:
                    conf_scores = det.boxes.conf.cpu().numpy()
                    # print(type(det.boxes.conf))
                    for idx, c_id in enumerate(list(det.boxes.cls.cpu().numpy())):
                        c = self.model.names[c_id]
                        if c not in detected_objects_with_scores:
                            detected_objects_with_scores[c] = conf_scores[idx]
                        else:
                            detected_objects_with_scores[c] = max(detected_objects_with_scores[c], conf_scores[idx].item())
                results.append(detected_objects_with_scores)
        return results

    
        