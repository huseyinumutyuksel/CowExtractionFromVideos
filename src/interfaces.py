from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np

class IDetector(ABC):
    """
    Interface for object detection models.
    """
    @abstractmethod
    def detect_and_track(self, frame: np.ndarray) -> List[Any]:
        """
        Detects and tracks objects in the given frame.
        Returns a list of results (format depends on implementation, but typically includes boxes and IDs).
        """
        pass

class IVideoProcessor(ABC):
    """
    Interface for video processing logic.
    """
    @abstractmethod
    def process_video(self, video_path: str):
        """
        Processes a single video file.
        """
        pass

class IWriterManager(ABC):
    """
    Interface for managing video writers for different tracks.
    """
    @abstractmethod
    def write_frame(self, track_id: int, frame: np.ndarray, fps: float):
        """
        Writes a frame to the video file corresponding to the track_id.
        """
        pass

    @abstractmethod
    def close_all(self):
        """
        Closes all open video writers.
        """
        pass
