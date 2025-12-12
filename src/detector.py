from typing import List, Any
import numpy as np
from ultralytics import YOLO
from src.interfaces import IDetector
import config.settings as settings

class YoloCowDetector(IDetector):
    def __init__(self, model_path: str = settings.YOLO_MODEL_NAME):
        self.model = YOLO(model_path)
        self.target_class_id = settings.TARGET_CLASS_ID
        self.conf_threshold = settings.CONFIDENCE_THRESHOLD

    def detect_and_track(self, frame: np.ndarray) -> List[Any]:
        # Persist=True is crucial for tracking to keep IDs consistent across frames
        results = self.model.track(
            frame, 
            persist=True, 
            verbose=False, 
            classes=[self.target_class_id],
            conf=self.conf_threshold
        )
        return results
