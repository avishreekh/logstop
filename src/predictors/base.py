from typing import List, Dict
import numpy as np

class LocalPropertyPredictor:
    """
    Base class for local property predictors.
    Given a sequence of items (e.g., video frames), this should return a trace of predictions for specified propositions.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
    
    def predict(self, frames: List[np.ndarray], local_properties: List[str], batch_size: int = 8) -> List[Dict[str, float]]:
        """
        Predict local properties for a batch of frames.
        Args:
            frames (List[np.ndarray]): List of frames as numpy arrays.
            local_properties (List[str]): List of local properties to predict.
            batch_size (int): Batch size for processing frames.
        Returns:
            List[Dict[str, float]]: A list where each element corresponds to a frame and contains a dictionary mapping local properties to their confidence scores.
                                    An example of the list for 2 frames could be: [{'local_property1': 0.95, 'local_property2': 0.80}, {'local_property1': 0.90}]

        """
        raise NotImplementedError
    
    def generate_trace(self, sequence_of_frames: List[np.ndarray], local_properties: List[str], batch_size: int = 8) -> Dict[str, list]:
        """
        Generate a trace for the given sequence of frames and propositions.
        Args:
            sequence_of_frames (List[np.ndarray]): List of frames as numpy arrays.
            local_properties (List[str]): List of local properties to detect.
            batch_size (int): Batch size for processing frames.
        Returns:
            Dict[str, list]: A trace dictionary mapping each local property to a list of scores over time.
                            An example trace for a sequence of 2 frames could be: {'local_property1': [0.95, 0.0], 'local_property2': [0.80, 0.0]}
        """
        trace = {p: [] for p in local_properties}
        batch_results = self.predict(sequence_of_frames, local_properties, batch_size=batch_size)
        for result in batch_results:
            for p in local_properties:
                score = result.get(p, 0.0)  # Default score is 0.0 if not detected
                trace[p].append(score)
        return trace
    
def get_local_property_predictor(model_path: str) -> LocalPropertyPredictor:
    """
    Factory function to get a local property predictor based on the model path.
    Args:
        model_path (str): Path to the model or identifier for the predictor.
    Returns:
        LocalPropertyPredictor: An instance of a LocalPropertyPredictor subclass.
    """
    if "yolo" in model_path.lower():
        from .object_detectors import YOLO
        return YOLO(model_path=model_path)
    else:
        raise ValueError(f"Unsupported local property predictor model: {model_path}")