import cv2
import glob
import os
import logging
import numpy as np
from tqdm import tqdm
from src.interfaces import IDetector, IWriterManager, IVideoProcessor
from src.smoother import BoxSmoother
import config.settings as settings

logger = logging.getLogger(__name__)

class CowExtractionProcessor(IVideoProcessor):
    def __init__(self, detector: IDetector, writer_manager: IWriterManager):
        self.detector = detector
        self.writer_manager = writer_manager
        self.smoother = BoxSmoother()

    def process_video(self, video_path: str):
        logger.info(f"Processing video: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Handle invalid FPS
        if fps <= 0 or np.isnan(fps):
            logger.warning(f"Invalid FPS {fps} in {os.path.basename(video_path)}, defaulting to 30.0")
            fps = 30.0
        else:
            # Round FPS to nearest integer to prevent ffmpeg timebase errors with weird floats
            # E.g. 240.37... -> 240
            fps = round(fps)
            logger.debug(f"Video FPS: {fps}")

        # Reset active writers and smoother for this new video
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        self.writer_manager.reset_track_mapping(video_stem)
        self.smoother.reset()
        
        # Get total frame count for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create progress bar for frame processing
        pbar = tqdm(total=total_frames, desc=f"Processing {video_stem}", unit="frame", leave=False)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use fixed resolution for consistent processing if needed, 
            # currently just processing whatever resolution the video is.

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
                        
                        # Apply padding to ensure we don't cut off edges (hooves, tails)
                        padding = getattr(settings, 'CROP_PADDING', 0)
                        x1 -= padding
                        y1 -= padding
                        x2 += padding
                        y2 += padding
                        
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
                                source_frame = self._apply_mask(frame, seg)
                        
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
            
            pbar.update(1)

        pbar.close()
        cap.release()
        self.writer_manager.close_all()

    def process_all_videos(self, skip_list=None):
        if skip_list is None:
            skip_list = []
            
        search_pattern = os.path.join(settings.INPUT_VIDEOS_DIR, f"*{settings.VIDEO_EXT}")
        video_files = glob.glob(search_pattern)
        
        logger.info(f"Found {len(video_files)} videos in {settings.INPUT_VIDEOS_DIR}")
        
        # Create progress bar for video-level progress
        pbar = tqdm(video_files, desc="Processing videos", unit="video")
        
        for video_file in pbar:
            # Check if video should be skipped
            if video_file in skip_list:
                logger.info(f"Skipping already processed video: {os.path.basename(video_file)}")
                continue
            
            pbar.set_description(f"Processing {os.path.basename(video_file)[:30]}")
            self.process_video(video_file)

        pbar.close()

        logger.info("Processing complete.")

    def _apply_mask(self, frame: np.ndarray, segment: np.ndarray) -> np.ndarray:
        """
        Applies background masking based on the configured method.
        """
        img_h, img_w = frame.shape[:2]
        method = getattr(settings, 'MASK_METHOD', 'soft')
        
        if method == 'binary':
            return self._apply_binary_mask(frame, segment, img_h, img_w)
        elif method == 'soft':
            return self._apply_soft_mask(frame, segment, img_h, img_w)
        else:
            print(f"Warning: Unknown mask method '{method}'. Defaulting to soft mask.")
            return self._apply_soft_mask(frame, segment, img_h, img_w)

    def _apply_binary_mask(self, frame: np.ndarray, segment: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Original hard-cut masking.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [segment.astype(np.int32)], 1)
        
        bg_color = settings.BACKGROUND_COLOR 
        background = np.full(frame.shape, bg_color, dtype=np.uint8)
        
        mask_bool = mask.astype(bool)
        return np.where(mask_bool[..., None], frame, background)

    def _apply_soft_mask(self, frame: np.ndarray, segment: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Soft masking with dilation and Gaussian blur for smoother edges.
        """
        # 1. Create base binary mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [segment.astype(np.int32)], 255) # Use 0-255 range

        # 2. Dilate to include potential edge pixels
        kernel = np.ones((3, 3), np.uint8)
        iterations = getattr(settings, 'MASK_DILATION_ITERATIONS', 2)
        mask = cv2.dilate(mask, kernel, iterations=iterations)

        # 3. Gaussian Blur for soft alpha
        blur_size = getattr(settings, 'MASK_BLUR_KERNEL_SIZE', (15, 15))
        mask_blurred = cv2.GaussianBlur(mask, blur_size, 0)
        
        # 4. Normalize alpha to 0.0 - 1.0
        alpha = mask_blurred.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=-1) # (H,W,1)

        # 5. Blend
        bg_color = settings.BACKGROUND_COLOR 
        background = np.full(frame.shape, bg_color, dtype=np.uint8)
        
        # Formula: Result = Foreground * Alpha + Background * (1 - Alpha)
        foreground = frame.astype(float)
        background = background.astype(float)
        
        out = (foreground * alpha + background * (1.0 - alpha))
        return out.astype(np.uint8)
