import os
import sys
import logging
import argparse
from src.detector import YoloCowDetector
from src.writer import CowVideoWriterManager
from src.processor import CowExtractionProcessor
import config.settings as settings
from src.scanner import VideoScanner
import glob
import shutil

def setup_logging(verbose=False, log_file=None):
    """
    Configure logging for the application.
    
    Args:
        verbose: If True, set log level to DEBUG
        log_file: Custom log file path (default: from settings)
    """
    log_level = logging.DEBUG if verbose else getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_path = log_file or settings.LOG_FILE
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        settings.LOG_FORMAT,
        datefmt=settings.LOG_DATE_FORMAT
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress overly verbose third-party loggers
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract and track individual cows from video files using YOLOv8.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process videos with default settings
  python main.py
  
  # Specify custom input/output directories
  python main.py --input-dir ./my_videos --output-dir ./my_output
  
  # Use a different YOLO model with higher confidence threshold
  python main.py --model yolov8l-seg.pt --confidence 0.85
  
  # Enable verbose logging and force clean start
  python main.py --verbose --clean
  
  # Resume processing from previous run
  python main.py --resume
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help=f'Input videos directory (default: {settings.INPUT_VIDEOS_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Output directory for extracted cow videos (default: {settings.OUTPUT_VIDEOS_DIR})'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=settings.YOLO_MODEL_NAME,
        help=f'YOLO model to use (default: {settings.YOLO_MODEL_NAME})'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=settings.CONFIDENCE_THRESHOLD,
        help=f'Detection confidence threshold 0.0-1.0 (default: {settings.CONFIDENCE_THRESHOLD})'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous run without prompting'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Delete existing output and start fresh'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG level) logging'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help=f'Custom log file path (default: {settings.LOG_FILE})'
    )
    
    parser.add_argument(
        '--no-scan',
        action='store_true',
        help='Skip pre-scan for single-cow videos'
    )
    
    return parser.parse_args()

def main():
    """Main application entry point."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose, log_file=args.log_file)
    logger.info("=" * 60)
    logger.info("Cow Extraction Project - Starting")
    logger.info("=" * 60)
    
    try:
        # Validate configuration
        logger.debug("Validating configuration...")
        settings.validate_config()
        logger.debug("Configuration validated successfully")
        
        # Override settings from CLI arguments
        if args.input_dir:
            settings.INPUT_VIDEOS_DIR = os.path.abspath(args.input_dir)
            logger.info(f"Input directory overridden: {settings.INPUT_VIDEOS_DIR}")
        
        if args.output_dir:
            settings.OUTPUT_VIDEOS_DIR = os.path.abspath(args.output_dir)
            logger.info(f"Output directory overridden: {settings.OUTPUT_VIDEOS_DIR}")
        
        if args.confidence != settings.CONFIDENCE_THRESHOLD:
            if not 0.0 <= args.confidence <= 1.0:
                raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {args.confidence}")
            settings.CONFIDENCE_THRESHOLD = args.confidence
            logger.info(f"Confidence threshold overridden: {settings.CONFIDENCE_THRESHOLD}")
        
        if args.model != settings.YOLO_MODEL_NAME:
            settings.YOLO_MODEL_NAME = args.model
            logger.info(f"YOLO model overridden: {settings.YOLO_MODEL_NAME}")
        
        # --- CHECK EXISTING OUTPUT ---
        processed_stems = set()
        should_delete = args.clean
        
        if os.path.exists(settings.OUTPUT_VIDEOS_DIR) and len(os.listdir(settings.OUTPUT_VIDEOS_DIR)) > 0:
            logger.info(f"Found existing output in: {settings.OUTPUT_VIDEOS_DIR}")
            
            if args.resume:
                should_delete = False
                logger.info("Resume mode enabled - will continue from previous run")
            elif not args.clean:
                # Interactive prompt
                while True:
                    try:
                        choice = input("Delete existing output and start over? (y/n): ").strip().lower()
                        if choice in ['y', 'yes']:
                            should_delete = True
                            break
                        elif choice in ['n', 'no']:
                            should_delete = False
                            break
                    except (EOFError, KeyboardInterrupt):
                        logger.warning("\nUser interrupted prompt, defaulting to resume mode")
                        should_delete = False
                        break
        
        # --- Cleanup Output Directory ---
        if should_delete:
            if os.path.exists(settings.OUTPUT_VIDEOS_DIR):
                logger.info(f"Cleaning previous output: {settings.OUTPUT_VIDEOS_DIR}")
                try:
                    shutil.rmtree(settings.OUTPUT_VIDEOS_DIR)
                    os.makedirs(settings.OUTPUT_VIDEOS_DIR, exist_ok=True)
                    logger.info("Output directory cleaned successfully")
                except OSError as e:
                    logger.error(f"Error cleaning output directory: {e}")
                    raise
        else:
            # Resume mode: find what's already done
            if os.path.exists(settings.OUTPUT_VIDEOS_DIR):
                logger.info("Scanning existing output to resume...")
                existing_files = os.listdir(settings.OUTPUT_VIDEOS_DIR)
                for f in existing_files:
                    if f.endswith(settings.VIDEO_EXT):
                        # Format: {source_stem}_cow_{counter}.mp4
                        parts = f.split('_cow_')
                        if len(parts) > 1:
                            source_stem = parts[0]
                            processed_stems.add(source_stem)
                logger.info(f"Found {len(processed_stems)} already processed videos.")
        
        # Ensure input directory exists
        if not os.path.exists(settings.INPUT_VIDEOS_DIR):
            logger.warning(f"Input directory '{settings.INPUT_VIDEOS_DIR}' does not exist.")
            logger.info("Creating empty input directory...")
            os.makedirs(settings.INPUT_VIDEOS_DIR, exist_ok=True)
            logger.info("Please add video files to the input directory and run again.")
            return
        
        # Initialize dependencies
        logger.info("Initializing YOLO detector...")
        detector = YoloCowDetector()
        
        # --- STEP 1: Scan for Single Cow Videos ---
        videos_to_scan = []
        processed_paths = []
        
        if not args.no_scan:
            logger.info("Initializing video scanner...")
            scanner = VideoScanner(detector)
            
            search_pattern = os.path.join(settings.INPUT_VIDEOS_DIR, f"*{settings.VIDEO_EXT}")
            all_videos = glob.glob(search_pattern)
            
            logger.info(f"Found {len(all_videos)} total videos to process")
            
            # Filter videos to scan: only those NOT processed yet
            for video_path in all_videos:
                stem = os.path.splitext(os.path.basename(video_path))[0]
                if stem in processed_stems:
                    processed_paths.append(video_path)
                    logger.debug(f"Skipping already processed: {stem}")
                else:
                    videos_to_scan.append(video_path)
            
            # Scan for single-cow videos
            if videos_to_scan:
                logger.info(f"Starting pre-scan for single-cow videos on {len(videos_to_scan)} videos...")
                scanner.scan_and_filter(videos_to_scan)
            else:
                logger.info("No new videos to scan")
        else:
            logger.info("Skipping single-cow video scan (--no-scan flag)")
            search_pattern = os.path.join(settings.INPUT_VIDEOS_DIR, f"*{settings.VIDEO_EXT}")
            all_videos = glob.glob(search_pattern)
            
            for video_path in all_videos:
                stem = os.path.splitext(os.path.basename(video_path))[0]
                if stem not in processed_stems:
                    videos_to_scan.append(video_path)
                else:
                    processed_paths.append(video_path)
        
        # --- STEP 2: Process the rest ---
        logger.info("Initializing video writer manager...")
        writer_manager = CowVideoWriterManager(settings.OUTPUT_VIDEOS_DIR)
        
        logger.info("Initializing video processor...")
        processor = CowExtractionProcessor(detector, writer_manager)
        
        # Process videos
        logger.info("Starting video processing...")
        processor.process_all_videos(skip_list=processed_paths)
        
        logger.info("=" * 60)
        logger.info("Cow Extraction Project - Completed Successfully")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
