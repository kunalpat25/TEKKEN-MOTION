import numpy as np
import cv2
import time
import imutils
import json
import os
from imutils.video import FileVideoStream, VideoStream


# Configuration constants
CAMERA_SOURCE = 0
FRAME_WIDTH = 600
FRAME_HEIGHT = 600
NOTIFICATION_COLOR = (0, 0, 255)
SETUP_COUNTDOWN = 7
POSE_COUNTDOWN = 7
TEXT_POSITION = (50, 50)
CONFIG_FILE_PATH = "gesture_config.json"


def calculate_center_point(rectangle):
    x, y, width, height = rectangle
    center_x = int(x + width // 2)
    center_y = int(y + height // 2)
    return (center_x, center_y)


def capture_frame_cv2(video_stream):
    success, frame = video_stream.read()
    if not success or frame is None:
        return None
    # Mirror for intuitive movement
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    return frame


def capture_frame_imutils(video_stream):
    frame = video_stream.read()
    if frame is None:
        return None
    # Mirror for intuitive movement
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    return frame


def render_rectangle(success, rectangle, image):
    if not success:
        return

    top_left = (int(rectangle[0]), int(rectangle[1]))
    bottom_right = (int(rectangle[0] + rectangle[2]), int(rectangle[1] + rectangle[3]))
    cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2, 1)


# Configuration file handling
def save_configuration(face_region, face_center, regions_count, gesture_regions):
    """Save configuration data to a JSON file"""
    config_data = {
        "face_region": list(face_region),
        "face_center": list(face_center),
        "regions_count": regions_count,
        "gesture_regions": [list(region) for region in gesture_regions],
    }

    with open(CONFIG_FILE_PATH, "w") as config_file:
        json.dump(config_data, config_file, indent=4)

    print(f"Configuration saved to {CONFIG_FILE_PATH}")


def load_configuration():
    """Load configuration data from a JSON file"""
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Configuration file {CONFIG_FILE_PATH} not found")
        return None

    with open(CONFIG_FILE_PATH, "r") as config_file:
        config_data = json.load(config_file)

    # Convert lists back to tuples for regions
    face_region = tuple(config_data["face_region"])
    face_center = tuple(config_data["face_center"])
    regions_count = config_data["regions_count"]
    gesture_regions = [tuple(region) for region in config_data["gesture_regions"]]

    return face_region, face_center, regions_count, gesture_regions


def configuration_exists():
    """Check if configuration file exists"""
    return os.path.exists(CONFIG_FILE_PATH)
