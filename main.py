import os
from src.detector import YoloCowDetector
from src.writer import CowVideoWriterManager
from src.processor import CowExtractionProcessor
import config.settings as settings

from src.scanner import VideoScanner
import glob

import shutil

def select_background_color():
    """
    Asks the user to select a background color from a list of ML-friendly options.
    Returns the selected color tuple (B, G, R).
    """
    colors = {
        "1": ("Yesil (Makine Ogrenmesi icin Onerilen)", (0, 255, 0)),
        "2": ("Siyah (Varsayilan)", (0, 0, 0)),
        "3": ("Beyaz", (255, 255, 255)),
        "4": ("Mavi (Chroma Key)", (255, 0, 0)),
        "5": ("Magenta (Yuksek Kontrast)", (255, 0, 255))
    }
    
    print("\n--- Arka Plan Rengi Secimi ---")
    print("Lutfen videolardaki nesnelerin (ineklerin) daha iyi ayirt edilebilmesi icin bir arka plan rengi secin.")
    for key, (name, _) in colors.items():
        print(f"{key}. {name}")
    
    choice = input("Seciminiz (1-5) [Varsayilan: 2]: ").strip()
    
    if choice in colors:
        selected_name, color_value = colors[choice]
        print(f"Secilen Renk: {selected_name}")
        return color_value
    else:
        print("Gecersiz secim veya bos birakildi. Varsayilan renk (Siyah) kullanilacak.")
        return colors["2"][1]

def main():
    # --- Background Color Selection ---
    # Update the settings with the user's choice before processing starts
    settings.BACKGROUND_COLOR = select_background_color()

    # --- CHECK EXISTING OUTPUT ---
    processed_stems = set()
    should_delete = False
    
    if os.path.exists(settings.OUTPUT_VIDEOS_DIR) and len(os.listdir(settings.OUTPUT_VIDEOS_DIR)) > 0:
        print(f"Found existing output in: {settings.OUTPUT_VIDEOS_DIR}")
        while True:
            choice = input("Delete existing output and start over? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                should_delete = True
                break
            elif choice in ['n', 'no']:
                should_delete = False
                break
    
    # --- Cleanup Output Directory ---
    if should_delete:
        if os.path.exists(settings.OUTPUT_VIDEOS_DIR):
            print(f"Cleaning previous output: {settings.OUTPUT_VIDEOS_DIR}")
            try:
                shutil.rmtree(settings.OUTPUT_VIDEOS_DIR)
                # Re-create immediately because other parts assume it exists
                os.makedirs(settings.OUTPUT_VIDEOS_DIR, exist_ok=True)
            except OSError as e:
                print(f"Error cleaning output directory: {e}")
    else:
        # Resume mode: find what's already done
        if os.path.exists(settings.OUTPUT_VIDEOS_DIR):
            print("Scanning existing output to resume...")
            existing_files = os.listdir(settings.OUTPUT_VIDEOS_DIR)
            for f in existing_files:
                if f.endswith(settings.VIDEO_EXT):
                    # Format: {source_stem}_cow_{counter}.mp4
                    # We need to extract {source_stem}
                    # Split by '_cow_' and take the first part
                    parts = f.split('_cow_')
                    if len(parts) > 1:
                        source_stem = parts[0]
                        processed_stems.add(source_stem)
            print(f"Found {len(processed_stems)} already processed videos.")

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
    
    # Filter videos to scan: only those NOT processed yet
    # We need to map stems back to full paths for the scanner
    videos_to_scan = []
    processed_paths = []
    
    for video_path in all_videos:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        if stem in processed_stems:
            processed_paths.append(video_path)
        else:
            videos_to_scan.append(video_path)
            
    # Identify and copy single-cow videos (but don't skip them in processing anymore!)
    print(f"Starting pre-scan for single-cow videos on {len(videos_to_scan)} videos...")
    scanner.scan_and_filter(videos_to_scan)
    
    # --- STEP 2: Process the rest ---
    print("Initializing Writer Manager...")
    writer_manager = CowVideoWriterManager(settings.OUTPUT_VIDEOS_DIR)
    
    print("Initializing Processor...")
    processor = CowExtractionProcessor(detector, writer_manager)
    
    # Run, skipping ONLY the videos that were already fully processed in a previous run
    # Single cow videos are NO LONGER skipped, they are processed.
    processor.process_all_videos(skip_list=processed_paths)

if __name__ == "__main__":
    main()
