import cv2
import os
import shutil
import logging
from typing import List, Set
from tqdm import tqdm
from src.interfaces import IDetector
import config.settings as settings

logger = logging.getLogger(__name__)

class VideoScanner:
    def __init__(self, detector: IDetector):
        self.detector = detector
        self.single_cow_dir = settings.SINGLE_COW_VIDEOS_DIR
        os.makedirs(self.single_cow_dir, exist_ok=True)

    def is_single_cow_video(self, video_path: str) -> bool:
        """
        Determines if a video contains exactly one unique cow track throughout its duration.
        Logic: 
        - If multiple cows appear simultaneously in any frame -> False.
        - If we see more than 1 unique track ID over the whole video -> False (conservative approach).
        - Actually, user said: "videos containing only one cow". This usually implies simultaneous presence.
          However, to be safe and strictly "single cow video", it should probably only ever have 1 cow.
          But YOLO track IDs might switch if tracking is lost. 
          Let's stick to: Max simultaneous detections == 1 AND at least 1 detection.
          If track ID switches, it's still 1 cow on screen, just re-identified.
          So: Max simultaneous cows <= 1 AND Total frames with cows > 0.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video for scanning: {video_path}")
            return False

        max_simultaneous_cows = 0
        frames_with_cows = 0
        
        # We can implement a frame skip to speed up scanning if needed, 
        # but for accuracy (finding 2 cows showing up briefly), scanning every frame is safer.
        # Given 105 videos, this might take a while.
        # Let's frame skip = 5 (check every 5th frame) to speed up. 
        # If a second cow appears for < 5 frames (<0.2s), it might be missed, but that's a reasonable trade-off.
        
        frame_idx = 0
        skip_frames = 5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % skip_frames == 0:
                results = self.detector.detect_and_track(frame)
                
                cow_count = 0
                if results and len(results) > 0:
                    res = results[0]
                    if res.boxes is not None and res.boxes.id is not None:
                        # Count valid boxes
                        cow_count = len(res.boxes)
                
                if cow_count > max_simultaneous_cows:
                    max_simultaneous_cows = cow_count
                
                if cow_count > 0:
                    frames_with_cows += 1
                
                # Early exit if we already found multiple cows
                if max_simultaneous_cows > 1:
                    break

            frame_idx += 1

        cap.release()

        # Logic for "Single Cow Video"
        # Must have seen at least one cow, and never more than one at a time.
        return max_simultaneous_cows == 1

    def scan_and_filter(self, video_files: List[str]) -> List[str]:
        """
        Scans videos. If single cow, copy to SINGLE_COW_VIDEOS_DIR and return as 'processed'.
        Returns a list of video paths that were identified as single-cow and processed.
        """
        logger.info(f"Starting scan of {len(video_files)} videos for single-cow filter...")
        single_cow_videos = []

        # Create progress bar for scanning
        pbar = tqdm(video_files, desc="Scanning videos", unit="video")
        
        for video_path in pbar:
            pbar.set_description(f"Scanning {os.path.basename(video_path)[:30]}")
            if self.is_single_cow_video(video_path):
                logger.info(f"Single cow video identified: {os.path.basename(video_path)}")
                
                # Copy original file to separate folder
                filename = os.path.basename(video_path)
                dest_path = os.path.join(self.single_cow_dir, filename)
                
                try:
                    shutil.copy2(video_path, dest_path)
                    single_cow_videos.append(video_path)
                    logger.debug(f"Copied to: {dest_path}")
                except OSError as e:
                    logger.error(f"Failed to copy {video_path}: {e}")
            else:
                logger.debug(f"Multi/No cow video: {os.path.basename(video_path)}")
        
        pbar.close()
        logger.info(f"Scan complete. Found {len(single_cow_videos)} single-cow videos.")
        return single_cow_videos
