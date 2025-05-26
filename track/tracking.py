import numpy as np
import cv2
import time

from .motion_detector import GestureController, get_configuration
from .utils import (
    CAMERA_SOURCE,
    NOTIFICATION_COLOR,
    capture_frame_cv2,
    render_rectangle,
    calculate_center_point,
    configuration_exists,
)
from .setup import initialize_configuration


def start_gesture_gaming():
    print("Initializing gesture-based game controls...")

    # Check if configuration exists, if not run setup first
    setup_was_run = False
    if not configuration_exists():
        print("No configuration found. Running setup first...")
        initialize_configuration()
        setup_was_run = True

    # Always get the latest configuration to ensure we're using the most recent settings
    tracking_data, tracking_center, _, _ = get_configuration()

    # Always create a fresh controller that will load the most recent configuration
    controller = GestureController(training_mode=False)
    tracking_area = controller.face_region  # Get the actual region from the controller

    face_tracker = cv2.TrackerCSRT_create()
    video_source = cv2.VideoCapture(CAMERA_SOURCE)

    initialization_time = 3
    start_time = time.time()

    while True:
        frame = capture_frame_cv2(video_source)
        elapsed_time = time.time() - start_time

        if elapsed_time > initialization_time or frame is None:
            break

        cv2.putText(
            frame,
            str(int(initialization_time - elapsed_time) + 1),
            (225, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            NOTIFICATION_COLOR,
            4,
        )
        cv2.putText(
            frame,
            "Keep your face inside the box",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            NOTIFICATION_COLOR,
            4,
        )
        render_rectangle(True, tracking_area, frame)
        cv2.imshow("Motion Tracking", frame)
        cv2.waitKey(1)

    face_tracker.init(frame, tracking_area)

    print("Tracking activated. Your movements will control the game.")
    print("Press Enter to exit.")

    while True:
        frame = capture_frame_cv2(video_source)
        if frame is None:
            break

        display_frame = frame.copy()

        success, current_region = face_tracker.update(frame)

        render_rectangle(True, current_region, frame)

        controller.process_frame(frame, calculate_center_point(current_region))

        cv2.imshow("Motion Tracking", frame)

        key_pressed = cv2.waitKey(1)
        if key_pressed == 13:  # Enter key
            break

    cv2.destroyAllWindows()
    video_source.release()
    print("Gesture control terminated.")


def main():
    start_gesture_gaming()


if __name__ == "__main__":
    main()
