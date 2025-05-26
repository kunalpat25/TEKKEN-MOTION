import numpy as np
import cv2
import time
import imutils

from track.motion_detector import GestureController
from track.utils import (
    CAMERA_SOURCE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    NOTIFICATION_COLOR,
    TEXT_POSITION,
    save_configuration,
)
from track.utils import capture_frame_cv2, render_rectangle, calculate_center_point


def initialize_configuration():
    print("Initializing configuration interface...")

    video_capture = cv2.VideoCapture(CAMERA_SOURCE)
    time.sleep(0.2)  # Allow camera to initialize

    preparation_time = 3.0
    start_time = time.time()

    while True:
        frame = capture_frame_cv2(video_capture)
        if frame is None:
            break

        frame = imutils.resize(frame, width=FRAME_WIDTH, height=FRAME_HEIGHT)
        elapsed_time = time.time() - start_time

        if elapsed_time > preparation_time:
            break

        cv2.putText(
            frame,
            str(int(preparation_time - elapsed_time) + 1),
            (225, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            NOTIFICATION_COLOR,
            4,
        )
        cv2.imshow("Configuration", frame)
        cv2.waitKey(1)

    reference_frame = frame.copy()
    cv2.destroyAllWindows()

    cv2.putText(
        frame,
        "Select facial tracking region",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        NOTIFICATION_COLOR,
        2,
    )
    face_region = cv2.selectROI(frame, False)
    face_center = calculate_center_point(face_region)

    cv2.destroyAllWindows()
    video_capture.release()

    print(f"Facial tracking region configured: {face_region}")

    gesture_controller = GestureController(
        face_region=face_region, regions_count=2, training_mode=True
    )

    # Save the configuration
    gesture_regions = gesture_controller.gesture_regions
    regions_count = len(gesture_regions)
    save_configuration(face_region, face_center, regions_count, gesture_regions)

    print("Configuration completed and saved. Ready to start gameplay.")
    return face_region, face_center, regions_count, gesture_regions


if __name__ == "__main__":
    initialize_configuration()
