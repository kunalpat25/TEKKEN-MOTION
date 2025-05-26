import cv2
import numpy as np
import json
import os
import imutils
import time

# Configuration constants
CAMERA_SOURCE = 0
FRAME_WIDTH = 600
FRAME_HEIGHT = 600
NOTIFICATION_COLOR = (0, 0, 255)
SETUP_COUNTDOWN = 3
CONFIG_FILE_PATH = "pose_config.json"
DEFAULT_COOLDOWN = 0.5  # seconds

# Default trigger zone (normalized coords relative to face)
# These now represent offsets from the face position
DEFAULT_TRIGGER_ZONE = {
    "x1_offset": 0.2,  # Starting 20% of screen width to the right of face
    "x2_offset": 0.6,  # Extending to 60% of screen width to the right of face
    "y1_offset": -0.3,  # Starting 30% of screen height above face
    "y2_offset": 0.3,  # Extending to 30% of screen height below face
    "is_relative": True,  # Flag to indicate this is a relative trigger zone
}


def capture_frame(video_stream):
    """Capture a frame from video stream with horizontal flipping"""
    success, frame = video_stream.read()
    if not success or frame is None:
        return None
    # Mirror for intuitive movement
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    return frame


def draw_trigger_zone(frame, trigger_zone, face_position=None):
    """Draw the trigger zone rectangle on the frame

    If face_position is provided and trigger_zone is relative, draws the zone relative to face.
    Otherwise draws it at absolute coordinates.
    """
    h, w, _ = frame.shape

    # Handle relative trigger zones that move with the face
    if face_position and trigger_zone.get("is_relative", False):
        face_x, face_y = face_position
        # Convert normalized face position to pixel coordinates
        face_x_px = int(face_x * w)
        face_y_px = int(face_y * h)

        # Calculate trigger zone coordinates relative to face position
        x1 = int(face_x_px + (trigger_zone["x1_offset"] * w))
        y1 = int(face_y_px + (trigger_zone["y1_offset"] * h))
        x2 = int(face_x_px + (trigger_zone["x2_offset"] * w))
        y2 = int(face_y_px + (trigger_zone["y2_offset"] * h))

        # Keep coordinates within frame bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw a line connecting face to trigger zone for visual reference
        cv2.line(frame, (face_x_px, face_y_px), (x1, y1), (0, 255, 255), 1)
    else:
        # Original absolute positioning for backward compatibility
        cv2.rectangle(
            frame,
            (int(trigger_zone["x1"] * w), int(trigger_zone["y1"] * h)),
            (int(trigger_zone["x2"] * w), int(trigger_zone["y2"] * h)),
            (0, 0, 255),
            2,
        )


def draw_landmarks(frame, landmarks, mp_pose):
    """Draw key pose landmarks on the frame"""
    h, w, _ = frame.shape
    if landmarks:
        # Draw left hand
        left_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]
        cx, cy = int(left_hand.x * w), int(left_hand.y * h)
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        # Draw right hand
        right_hand = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
        cx, cy = int(right_hand.x * w), int(right_hand.y * h)
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        # Draw nose
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        cx, cy = int(nose.x * w), int(nose.y * h)
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        # Draw knees
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        cx1, cy1 = int(left_knee.x * w), int(left_knee.y * h)
        cx2, cy2 = int(right_knee.x * w), int(right_knee.y * h)
        cv2.circle(frame, (cx1, cy1), 10, (0, 255, 0), -1)
        cv2.circle(frame, (cx2, cy2), 10, (0, 255, 0), -1)

        # Draw shoulders
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        cx1, cy1 = int(left_shoulder.x * w), int(left_shoulder.y * h)
        cx2, cy2 = int(right_shoulder.x * w), int(right_shoulder.y * h)
        cv2.circle(frame, (cx1, cy1), 5, (0, 255, 0), -1)
        cv2.circle(frame, (cx2, cy2), 5, (0, 255, 0), -1)
        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)


def save_pose_configuration(trigger_zone, cooldown=DEFAULT_COOLDOWN):
    """Save pose configuration data to a JSON file"""
    config_data = {"trigger_zone": trigger_zone, "cooldown": cooldown}

    with open(CONFIG_FILE_PATH, "w") as config_file:
        json.dump(config_data, config_file, indent=4)

    print(f"Pose configuration saved to {CONFIG_FILE_PATH}")


def load_pose_configuration():
    """Load pose configuration data from a JSON file"""
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Configuration file {CONFIG_FILE_PATH} not found, using defaults")
        return DEFAULT_TRIGGER_ZONE, DEFAULT_COOLDOWN

    with open(CONFIG_FILE_PATH, "r") as config_file:
        config_data = json.load(config_file)

    trigger_zone = config_data.get("trigger_zone", DEFAULT_TRIGGER_ZONE)

    # Handle conversion from old format to new format if needed
    if not trigger_zone.get("is_relative", False) and all(
        k in trigger_zone for k in ["x1", "x2", "y1", "y2"]
    ):
        print("Converting absolute trigger zone to relative format")
        # Center of screen as default face position for conversion
        center_x, center_y = 0.5, 0.5
        trigger_zone = {
            "x1_offset": trigger_zone["x1"] - center_x,
            "x2_offset": trigger_zone["x2"] - center_x,
            "y1_offset": trigger_zone["y1"] - center_y,
            "y2_offset": trigger_zone["y2"] - center_y,
            "is_relative": True,
        }
        # Save the conversion for future use
        save_pose_configuration(
            trigger_zone, config_data.get("cooldown", DEFAULT_COOLDOWN)
        )

    cooldown = config_data.get("cooldown", DEFAULT_COOLDOWN)

    return trigger_zone, cooldown


def configuration_exists():
    """Check if pose configuration file exists"""
    return os.path.exists(CONFIG_FILE_PATH)


def point_in_trigger_zone(point_x, point_y, trigger_zone, face_position=None):
    """Check if a point (normalized coordinates) is inside the trigger zone

    If face_position is provided and trigger_zone is relative,
    calculates the zone position relative to the face.
    """
    # Handle relative trigger zones that move with the face
    if face_position and trigger_zone.get("is_relative", False):
        face_x, face_y = face_position

        # Calculate boundaries relative to face position
        zone_x1 = face_x + trigger_zone["x1_offset"]
        zone_x2 = face_x + trigger_zone["x2_offset"]
        zone_y1 = face_y + trigger_zone["y1_offset"]
        zone_y2 = face_y + trigger_zone["y2_offset"]

        # Clamp values to stay within normalized coordinates (0-1)
        zone_x1 = max(0.0, min(zone_x1, 1.0))
        zone_x2 = max(0.0, min(zone_x2, 1.0))
        zone_y1 = max(0.0, min(zone_y1, 1.0))
        zone_y2 = max(0.0, min(zone_y2, 1.0))

        return zone_x1 <= point_x <= zone_x2 and zone_y1 <= point_y <= zone_y2
    else:
        # Original absolute positioning for backward compatibility
        return (
            trigger_zone["x1"] <= point_x <= trigger_zone["x2"]
            and trigger_zone["y1"] <= point_y <= trigger_zone["y2"]
        )
