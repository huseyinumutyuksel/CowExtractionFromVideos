import sys
import os
from src.detector import YoloCowDetector
from src.writer import CowVideoWriterManager
from src.processor import CowExtractionProcessor
import config.settings as settings

from src.scanner import VideoScanner
import glob

def main():
    # Ensure input directory exists or warn
    if not os.path.exists(settings.INPUT_VIDEOS_DIR):
        print(f"WARNING: Input directory '{settings.INPUT_VIDEOS_DIR}' does not exist.")
        print("Please create it and put the videos there, or update config/settings.py")
        # We can try to create it to be helpful
        os.makedirs(settings.INPUT_VIDEOS_DIR, exist_ok=True)
        print("Created empty input directory.")
    
    # Initialize dependencies
    print("Initializing YOLO detector...")
    detector = YoloCowDetector() # Download will happen here if needed
    
    # --- STEP 1: Scan for Single Cow Videos ---
    print("Initializing Scanner...")
    scanner = VideoScanner(detector)
    
    search_pattern = os.path.join(settings.INPUT_VIDEOS_DIR, f"*{settings.VIDEO_EXT}")
    all_videos = glob.glob(search_pattern)
    
    # Identify and copy single-cow videos
    print("Starting pre-scan for single-cow videos...")
    single_cow_videos = scanner.scan_and_filter(all_videos)
    
    # --- STEP 2: Process the rest ---
    print("Initializing Writer Manager...")
    writer_manager = CowVideoWriterManager(settings.OUTPUT_VIDEOS_DIR)
    
    print("Initializing Processor...")
    processor = CowExtractionProcessor(detector, writer_manager)
    
    # Run, skipping the single-cow videos found above
    processor.process_all_videos(skip_list=single_cow_videos)

if __name__ == "__main__":
    main()
