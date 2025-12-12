import cv2
import glob
import os
import numpy as np
from src.interfaces import IDetector, IWriterManager, IVideoProcessor
import config.settings as settings

class CowExtractionProcessor(IVideoProcessor):
    def __init__(self, detector: IDetector, writer_manager: IWriterManager):
        self.detector = detector
        self.writer_manager = writer_manager

    def process_video(self, video_path: str):
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Handle invalid FPS
        if fps <= 0 or np.isnan(fps):
            print(f"Warning: Invalid FPS {fps}, defaulting to 30.0")
            fps = 30.0

        # Reset active writers for this new video
        self.writer_manager.reset_track_mapping()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.detector.detect_and_track(frame)
            
            if results and len(results) > 0:
                res = results[0]
                
                if res.boxes is not None and res.boxes.id is not None:
                    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                    ids = res.boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, ids):
                        x1, y1, x2, y2 = box
                        
                        # Ensure coordinates are within frame
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        cow_crop = frame[y1:y2, x1:x2]
                        
                        if cow_crop.size == 0:
                            continue

                        # Standardize resolution
                        target_w, target_h = settings.OUTPUT_RESOLUTION
                        h, w = cow_crop.shape[:2]
                        
                        if (w, h) != (target_w, target_h):
                            cow_crop = cv2.resize(cow_crop, (target_w, target_h))
                        
                        self.writer_manager.write_frame(track_id, cow_crop, fps)

        cap.release()
        self.writer_manager.close_all()

    def process_all_videos(self):
        search_pattern = os.path.join(settings.INPUT_VIDEOS_DIR, f"*{settings.VIDEO_EXT}")
        video_files = glob.glob(search_pattern)
        
        print(f"Found {len(video_files)} videos in {settings.INPUT_VIDEOS_DIR}")
        
        for video_file in video_files:
            self.process_video(video_file)

        print("Processing complete.")
