import os
import sys

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input settings
# Can be overridden via environment variables or CLI arguments
# Default: 'input_videos' folder in project root
INPUT_VIDEOS_DIR = os.getenv('COW_INPUT_DIR', os.path.join(BASE_DIR, 'input_videos'))

# Output settings
# Extracted cow videos will be saved here
# Default: 'output_cows' folder in project root
OUTPUT_VIDEOS_DIR = os.getenv('COW_OUTPUT_DIR', os.path.join(BASE_DIR, 'output_cows'))

# Single cow videos (identified by scanner) will be copied here
# Default: 'single_cow_videos' folder in project root
SINGLE_COW_VIDEOS_DIR = os.getenv('COW_SINGLE_DIR', os.path.join(BASE_DIR, 'single_cow_videos'))

# Smoothing settings
SMOOTHING_ALPHA = 0.2  # Lower = smoother but more lag (0.0 to 1.0)

# Model settings
# YOLO model to use (yolov8n.pt, yolov8s.pt, etc. will be downloaded automatically if not present)
#YOLO_MODEL_NAME = 'yolov8n.pt'
YOLO_MODEL_NAME = 'yolov8m-seg.pt'

# Processing settings
BORDER_MARGIN = 5
CONFIDENCE_THRESHOLD = 0.75
TARGET_CLASS_ID = 19  # COCO class id for 'cow' is 19. Change if using a custom model.
BACKGROUND_COLOR = (0, 0, 0)  # Black background to minimize distortion visibility

# Masking settings
MASK_METHOD = 'soft'  # Options: 'binary', 'soft'
MASK_BLUR_KERNEL_SIZE = (15, 15)  # Kernel size for Gaussian blur (must be odd numbers)
MASK_DILATION_ITERATIONS = 2      # Number of iterations to dilate the mask before blurring

# File extension for input and output
VIDEO_EXT = '.mp4'

# Output configurations
OUTPUT_RESOLUTION = (640, 640)  # Width, Height
MIN_TRACK_DURATION_SEC = 4.0
CROP_PADDING = 30  # Extra pixels around the detection box to prevent clipping (e.g. hooves, tails)

# Logging settings
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'cow_extraction.log')
LOG_LEVEL = os.getenv('COW_LOG_LEVEL', 'INFO')  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Create output and log directories if they don't exist
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def validate_config():
    """
    Validates configuration settings.
    Raises ValueError if any setting is invalid.
    """
    errors = []
    
    # Validate SMOOTHING_ALPHA
    if not 0.0 <= SMOOTHING_ALPHA <= 1.0:
        errors.append(f"SMOOTHING_ALPHA must be between 0.0 and 1.0, got {SMOOTHING_ALPHA}")
    
    # Validate CONFIDENCE_THRESHOLD
    if not 0.0 <= CONFIDENCE_THRESHOLD <= 1.0:
        errors.append(f"CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got {CONFIDENCE_THRESHOLD}")
    
    # Validate OUTPUT_RESOLUTION
    if not isinstance(OUTPUT_RESOLUTION, tuple) or len(OUTPUT_RESOLUTION) != 2:
        errors.append(f"OUTPUT_RESOLUTION must be a tuple of (width, height), got {OUTPUT_RESOLUTION}")
    elif any(v <= 0 for v in OUTPUT_RESOLUTION):
        errors.append(f"OUTPUT_RESOLUTION dimensions must be positive, got {OUTPUT_RESOLUTION}")
    
    # Validate MIN_TRACK_DURATION_SEC
    if MIN_TRACK_DURATION_SEC < 0:
        errors.append(f"MIN_TRACK_DURATION_SEC must be non-negative, got {MIN_TRACK_DURATION_SEC}")
    
    # Validate MASK_METHOD
    if MASK_METHOD not in ['binary', 'soft']:
        errors.append(f"MASK_METHOD must be 'binary' or 'soft', got '{MASK_METHOD}'")
    
    # Validate MASK_BLUR_KERNEL_SIZE
    if not isinstance(MASK_BLUR_KERNEL_SIZE, tuple) or len(MASK_BLUR_KERNEL_SIZE) != 2:
        errors.append(f"MASK_BLUR_KERNEL_SIZE must be a tuple of (width, height)")
    elif any(v % 2 == 0 or v <= 0 for v in MASK_BLUR_KERNEL_SIZE):
        errors.append(f"MASK_BLUR_KERNEL_SIZE values must be positive odd numbers, got {MASK_BLUR_KERNEL_SIZE}")
    
    # Validate BACKGROUND_COLOR
    if not isinstance(BACKGROUND_COLOR, tuple) or len(BACKGROUND_COLOR) != 3:
        errors.append(f"BACKGROUND_COLOR must be a tuple of (R, G, B)")
    elif any(not 0 <= v <= 255 for v in BACKGROUND_COLOR):
        errors.append(f"BACKGROUND_COLOR values must be between 0 and 255, got {BACKGROUND_COLOR}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors))
    
    return True
