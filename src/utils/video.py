"""
Utility functions for video processing.
"""
import cv2
from typing import List
import numpy as np

def extract_frames_from_video(video_path: str, fps: int = 30) -> List[np.ndarray]:
    """
    Extract frames from a video at the specified frames per second (fps).
    Args:
        video_path (str): Path to the input video file.
        fps (int): Frames per second to extract.
    Returns:
        List[np.ndarray]: List of extracted frames as numpy arrays.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    if frame_interval == 0:
        frame_interval = 1

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames