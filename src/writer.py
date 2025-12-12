import cv2
import os
import shutil
import numpy as np
from src.interfaces import IWriterManager
import config.settings as settings

class TrackInfo:
    def __init__(self, writer, temp_path, fps):
        self.writer = writer
        self.temp_path = temp_path
        self.fps = fps
        self.frame_count = 0

class CowVideoWriterManager(IWriterManager):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.global_cow_counter = 0 
        # Using a dictionary to store info about active tracks
        # track_id -> TrackInfo
        self.current_video_writers = {} 
        
        # Ensure output dir exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_next_filename(self) -> str:
        self.global_cow_counter += 1
        filename = f"cow_{self.global_cow_counter:04d}{settings.VIDEO_EXT}"
        return os.path.join(self.output_dir, filename)

    def write_frame(self, track_id: int, frame: np.ndarray, fps: float):
        height, width = frame.shape[:2]
        
        if track_id not in self.current_video_writers:
            # Create new writer with temp name in output dir to ensure same filesystem for easy move
            # temp_{track_id}_{random}.mp4
            temp_filename = f"temp_{track_id}_{os.urandom(4).hex()}{settings.VIDEO_EXT}"
            output_path = os.path.join(self.output_dir, temp_filename)
            
            # Use provided fps
            if fps <= 0:
                fps = 30.0 # Fallback
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            self.current_video_writers[track_id] = TrackInfo(writer, output_path, fps)
        
        track_info = self.current_video_writers[track_id]
        track_info.writer.write(frame)
        track_info.frame_count += 1

    def close_all(self):
        # Iterate over all current writers
        # Release them
        # Check duration
        # Rename or Delete
        
        # We need to collect keys first to avoid modification issues if any (though we are clearing at end)
        for trk_id, track_info in self.current_video_writers.items():
            track_info.writer.release()
            
            duration = track_info.frame_count / track_info.fps if track_info.fps > 0 else 0
            
            if duration >= settings.MIN_TRACK_DURATION_SEC:
                # Sufficient duration, finalize the file
                final_path = self.get_next_filename()
                
                # If file exists, remove it (overwrite logic for sequential purity in this session)
                if os.path.exists(final_path):
                    try:
                        os.remove(final_path)
                    except OSError:
                        pass # Best effort
                
                try:
                    shutil.move(track_info.temp_path, final_path)
                    # print(f"Saved cow track {trk_id} ({duration:.2f}s) -> {os.path.basename(final_path)}")
                except OSError as e:
                    print(f"Error renaming temp file {track_info.temp_path}: {e}")
            else:
                # Too short, discard
                # print(f"Discarded cow track {trk_id} ({duration:.2f}s < {settings.MIN_TRACK_DURATION_SEC}s)")
                if os.path.exists(track_info.temp_path):
                    try:
                        os.remove(track_info.temp_path)
                    except OSError:
                        pass

        self.current_video_writers.clear()

    def reset_track_mapping(self):
        """Call this between source videos."""
        self.close_all()
