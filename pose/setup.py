import cv2
import mediapipe as mp
import time
import numpy as np

from .utils import (
    CAMERA_SOURCE,
    NOTIFICATION_COLOR,
    SETUP_COUNTDOWN,
    DEFAULT_TRIGGER_ZONE,
    DEFAULT_COOLDOWN,
    capture_frame,
    draw_trigger_zone,
    draw_landmarks,
    save_pose_configuration,
)


def initialize_pose_configuration():
    """Interactive setup for pose-based gesture control"""
    print("Initializing pose gesture configuration...")

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None, DEFAULT_COOLDOWN

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initial countdown to prepare user
    start_time = time.time()
    while True:
        frame = capture_frame(cap)
        if frame is None:
            break

        elapsed_time = time.time() - start_time
        if elapsed_time > SETUP_COUNTDOWN:
            break

        # Display countdown
        cv2.putText(
            frame,
            f"Preparing pose detection: {int(SETUP_COUNTDOWN - elapsed_time) + 1}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            NOTIFICATION_COLOR,
            2,
        )
        cv2.imshow("Pose Configuration", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            cap.release()
            cv2.destroyAllWindows()
            return None, DEFAULT_COOLDOWN

    # Show current pose detection
    print("Showing pose detection. Position yourself in frame.")
    print("You will be able to define a trigger zone in the next step.")

    display_time = 3.0
    start_time = time.time()

    # Track face position to show how the trigger zone will move
    face_position = (0.5, 0.5)  # Default to center if no face detected
    trigger_zone = DEFAULT_TRIGGER_ZONE.copy()

    while time.time() - start_time < display_time:
        frame = capture_frame(cap)
        if frame is None:
            break

        # Convert to RGB and process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Draw key landmarks
            draw_landmarks(frame, results.pose_landmarks.landmark, mp_pose)
            # Update face position from nose landmark
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            face_position = (nose.x, nose.y)

        # Show temporary trigger zone that moves with face
        draw_trigger_zone(frame, trigger_zone, face_position)

        cv2.putText(
            frame,
            "Notice how trigger zone moves with your face",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            NOTIFICATION_COLOR,
            2,
        )

        cv2.imshow("Pose Configuration", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            cap.release()
            cv2.destroyAllWindows()
            return None, DEFAULT_COOLDOWN

    # Instructions for setting trigger zone
    print("\nDefine your trigger zone RELATIVE to your face position:")
    print("The zone will always maintain this relative position as you move.")
    print("1. Position yourself naturally in the camera")
    print("2. Click and drag to define the rectangular trigger zone")
    print("3. Press 'Enter' to confirm or 'c' to cancel and try again\n")

    # Capture one more frame for the ROI selection
    frame = capture_frame(cap)
    if frame is None:
        print("Error: Could not capture frame for ROI selection.")
        cap.release()
        cv2.destroyAllWindows()
        return None, DEFAULT_COOLDOWN

    # Process this frame to get face position
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        face_position = (nose.x, nose.y)
        # Draw face marker to help user visualize the reference point
        h, w, _ = frame.shape
        face_x_px, face_y_px = int(nose.x * w), int(nose.y * h)
        cv2.circle(frame, (face_x_px, face_y_px), 10, (255, 0, 0), -1)
        cv2.putText(
            frame,
            "Face Reference Point",
            (face_x_px + 15, face_y_px),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )
    else:
        print("Warning: Face not detected. Using center of frame as reference.")
        face_position = (0.5, 0.5)
        h, w, _ = frame.shape
        face_x_px, face_y_px = int(w / 2), int(h / 2)
        cv2.circle(frame, (face_x_px, face_y_px), 10, (255, 0, 0), -1)

    # Let user select the ROI (region of interest)
    roi = cv2.selectROI("Define Trigger Zone Relative to Face", frame, False)
    cv2.destroyWindow("Define Trigger Zone Relative to Face")

    if sum(roi) == 0:
        print("ROI selection canceled. Using default relative trigger zone.")
        trigger_zone = DEFAULT_TRIGGER_ZONE.copy()
    else:
        # Convert ROI to offsets relative to face position
        h, w, _ = frame.shape
        face_x_px, face_y_px = int(face_position[0] * w), int(face_position[1] * h)

        # Calculate offsets in normalized coordinates
        trigger_zone = {
            "x1_offset": (roi[0] - face_x_px) / w,
            "y1_offset": (roi[1] - face_y_px) / h,
            "x2_offset": (roi[0] + roi[2] - face_x_px) / w,
            "y2_offset": (roi[1] + roi[3] - face_y_px) / h,
            "is_relative": True,
        }
        print(f"Relative trigger zone set to: {trigger_zone}")

    # Ask for cooldown duration
    cooldown = DEFAULT_COOLDOWN
    try:
        cooldown_input = input(
            f"\nEnter cooldown duration in seconds (default is {DEFAULT_COOLDOWN}s): "
        )
        if cooldown_input.strip():
            cooldown = float(cooldown_input)
            if cooldown < 0.1:
                print("Cooldown too low, setting to 0.1s")
                cooldown = 0.1
            elif cooldown > 2.0:
                print("Cooldown too high, setting to 2.0s")
                cooldown = 2.0
    except ValueError:
        print(f"Invalid input, using default cooldown: {DEFAULT_COOLDOWN}s")
        cooldown = DEFAULT_COOLDOWN

    # Show the final configuration
    print("\nConfiguration Summary:")
    print(f"Relative Trigger Zone: {trigger_zone}")
    print(f"Cooldown: {cooldown}s")

    # Save the configuration
    save_pose_configuration(trigger_zone, cooldown)

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    return trigger_zone, cooldown


if __name__ == "__main__":
    initialize_pose_configuration()
