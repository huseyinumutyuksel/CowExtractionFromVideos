import sys
import os
from src.detector import YoloCowDetector
from src.writer import CowVideoWriterManager
from src.processor import CowExtractionProcessor
import config.settings as settings

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
    
    print("Initializing Writer Manager...")
    writer_manager = CowVideoWriterManager(settings.OUTPUT_VIDEOS_DIR)
    
    print("Initializing Processor...")
    processor = CowExtractionProcessor(detector, writer_manager)
    
    # Run
    processor.process_all_videos()

if __name__ == "__main__":
    main()
