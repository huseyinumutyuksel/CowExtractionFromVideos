import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input settings
# Change this path to the folder where your 105 videos are located
INPUT_VIDEOS_DIR = os.path.join(BASE_DIR, 'C:\\Users\\Umut\\Desktop\\cow_videos')

# Output settings
# Extracted cow videos will be saved here
OUTPUT_VIDEOS_DIR = os.path.join(BASE_DIR, 'C:\\Users\\Umut\\Desktop\\cow_single_videos')
SINGLE_COW_VIDEOS_DIR = os.path.join(BASE_DIR, 'C:\\Users\\Umut\\Desktop\\raw_single_cow_videos')

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

# File extension for input and output
VIDEO_EXT = '.mp4'

# Output configurations
OUTPUT_RESOLUTION = (640, 640)  # Width, Height
MIN_TRACK_DURATION_SEC = 4.0

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
