from typing import List, Any
import logging
import numpy as np
from ultralytics import YOLO
from src.interfaces import IDetector
import config.settings as settings

logger = logging.getLogger(__name__)

class YoloCowDetector(IDetector):
    def __init__(self, model_path: str = settings.YOLO_MODEL_NAME):
        logger.info(f"Initializing YOLO detector with model: {model_path}")
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        self.target_class_id = settings.TARGET_CLASS_ID
        self.conf_threshold = settings.CONFIDENCE_THRESHOLD
        logger.debug(f"Target class ID: {self.target_class_id}, Confidence threshold: {self.conf_threshold}")

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
