import numpy as np
import config.settings as settings

class BoxSmoother:
    def __init__(self, alpha: float = None):
        """
        alpha: Smoothing factor between 0 and 1.
               Low alpha = more smoothing (slower response to change, less jitter).
               High alpha = less smoothing (faster response, more jitter).
               Default is settings.SMOOTHING_ALPHA.
        """
        self.alpha = alpha if alpha is not None else settings.SMOOTHING_ALPHA
        self.tracks = {} # track_id -> [x1, y1, x2, y2] (float)

    def update(self, track_id: int, box: list) -> list:
        """
        Update the smoothed box for the given track_id with the new observation `box`.
        Returns the smoothed box as [x1, y1, x2, y2] (integers).
        """
        current_box = np.array(box, dtype=float)
        
        if track_id not in self.tracks:
            # First observation, initialize
            self.tracks[track_id] = current_box
            smoothed_box = current_box
        else:
            # EMA formula: smoothed = alpha * new + (1 - alpha) * old
            prev_box = self.tracks[track_id]
            smoothed_box = self.alpha * current_box + (1 - self.alpha) * prev_box
            self.tracks[track_id] = smoothed_box
            
        return smoothed_box.astype(int).tolist()

    def reset(self):
        self.tracks.clear()
