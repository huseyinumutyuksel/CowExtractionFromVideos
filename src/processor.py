import cv2
import glob
import os
import numpy as np
from src.interfaces import IDetector, IWriterManager, IVideoProcessor
from src.smoother import BoxSmoother
import config.settings as settings

class CowExtractionProcessor(IVideoProcessor):
    def __init__(self, detector: IDetector, writer_manager: IWriterManager):
        self.detector = detector
        self.writer_manager = writer_manager
        self.smoother = BoxSmoother()

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
        else:
            # Round FPS to nearest integer to prevent ffmpeg timebase errors with weird floats
            # E.g. 240.37... -> 240
            fps = round(fps)

        # Reset active writers and smoother for this new video
        self.writer_manager.reset_track_mapping()
        self.smoother.reset()

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
                    
                    # Get masks if available
                    segments = None
                    if res.masks is not None:
                        segments = res.masks.xy

                    for i, (box, track_id) in enumerate(zip(boxes, ids)):
                        # -------------------------
                        # 1. Partial Cow Filter
                        # -------------------------
                        raw_x1, raw_y1, raw_x2, raw_y2 = box
                        img_h, img_w = frame.shape[:2]
                        margin = settings.BORDER_MARGIN

                        # Check if box touches border (using raw detection to be safe)
                        if (raw_x1 <= margin) or (raw_y1 <= margin) or (raw_x2 >= img_w - margin) or (raw_y2 >= img_h - margin):
                            continue

                        # Apply smoothing to the box
                        box = self.smoother.update(track_id, box)
                        
                        x1, y1, x2, y2 = box
                        
                        # Ensure coordinates are within frame
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)

                        # -------------------------
                        # 2. Background Removal
                        # -------------------------
                        # Default to original frame
                        source_frame = frame
                        
                        # Apply mask if available
                        if segments is not None and len(segments) > i:
                            seg = segments[i]
                            if seg is not None and len(seg) > 0:
                                # Create a black mask of the same size as the frame
                                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                                # Fill the polygon (segment) with white (255)
                                cv2.fillPoly(mask, [seg.astype(np.int32)], 255)
                                
                                # Apply the mask to the frame (bitwise AND)
                                # Everything outside the mask becomes black
                                source_frame = cv2.bitwise_and(frame, frame, mask=mask)
                        
                        cow_crop = source_frame[y1:y2, x1:x2]
                        
                        if cow_crop.size == 0:
                            continue

                        # Standardize resolution with PADDING (Letterboxing) to prevent distortion
                        target_w, target_h = settings.OUTPUT_RESOLUTION
                        h, w = cow_crop.shape[:2]
                        
                        # Create black canvas
                        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                        
                        # Scaling logic: Only scale DOWN if crop is larger than target
                        # Otherwise keep original size to avoid "zoom"
                        scale = 1.0
                        if w > target_w or h > target_h:
                            scale = min(target_w / w, target_h / h)
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            cow_crop = cv2.resize(cow_crop, (new_w, new_h))
                            h, w = new_h, new_w # Update dims after resize
                        
                        # Calculate centering position
                        x_offset = (target_w - w) // 2
                        y_offset = (target_h - h) // 2
                        
                        # Place crop on canvas
                        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cow_crop
                        
                        # Use canvas as the frame to write
                        cow_crop = canvas
                        
                        self.writer_manager.write_frame(track_id, cow_crop, fps)

        cap.release()
        self.writer_manager.close_all()

    def process_all_videos(self, skip_list=None):
        if skip_list is None:
            skip_list = []
            
        search_pattern = os.path.join(settings.INPUT_VIDEOS_DIR, f"*{settings.VIDEO_EXT}")
        video_files = glob.glob(search_pattern)
        
        print(f"Found {len(video_files)} videos in {settings.INPUT_VIDEOS_DIR}")
        
        for video_file in video_files:
            # Check if video should be skipped
            if video_file in skip_list:
                print(f"Skipping single-cow video: {os.path.basename(video_file)}")
                continue
                
            self.process_video(video_file)

        print("Processing complete.")
