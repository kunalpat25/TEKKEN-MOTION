import cv2
import mediapipe as mp
from pynput.keyboard import Controller
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.input import simulate_key_press, simulate_key_release

from pose.utils import (
    capture_frame,
    draw_trigger_zone,
    draw_landmarks,
    load_pose_configuration,
    configuration_exists,
    point_in_trigger_zone,
)
from .setup import initialize_pose_configuration
from core.settings import KEY_MAPPINGS


class PoseGestureController:
    def __init__(self):
        self.keyboard = Controller()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.cap = None

        # Load or initialize configuration
        if not configuration_exists():
            print("No pose configuration found. Running setup...")
            self.trigger_zone, self.cooldown = initialize_pose_configuration()
        else:
            self.trigger_zone, self.cooldown = load_pose_configuration()

        self.last_action_time = 0
        self.face_position = (0.5, 0.5)  # Default to center if no face detected

        # Action cooldown timers
        self.action_cooldowns = {
            "punch": 0,
            "kick": 0,
            "power_move": 0,
        }

        # Movement control parameters
        self.previous_shoulders_position = None
        self.reference_shoulders_position = None
        self.horizontal_key = None
        self.horizontal_movement_threshold = 0.02  # Normalized coordinates
        self.horizontal_sensitivity = 0.05  # Normalized coordinates

        # Time tracking for movement updates
        self.last_movement_update = time.time()
        self.movement_update_interval = 0.1  # seconds

    def trigger_action(self, action_type="punch"):
        """Trigger an action based on key mappings

        Args:
            action_type: Type of action to trigger (punch, kick, etc.)
        """
        key = KEY_MAPPINGS.get(
            action_type, KEY_MAPPINGS["punch"]
        )  # Default to punch if action not found
        simulate_key_press(key)
        time.sleep(0.01)  # Small delay to ensure key press is registered
        simulate_key_release(key)
        print(f"{action_type.capitalize()} triggered")
        self.last_action_time = time.time()

    def process_movement(self, landmarks):
        """Process user movement based on shoulder positions

        Args:
            landmarks: MediaPipe pose landmarks
        """
        now = time.time()
        if now - self.last_movement_update < self.movement_update_interval:
            return

        self.last_movement_update = now

        # Get shoulder landmarks for horizontal movement
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate center point between shoulders
        shoulders_x = (left_shoulder.x + right_shoulder.x) / 2

        # Initialize reference position if needed
        if (
            self.reference_shoulders_position is None
            or self.previous_shoulders_position is None
        ):
            self.reference_shoulders_position = shoulders_x
            self.previous_shoulders_position = shoulders_x
            return

        # Calculate movement delta
        delta_x = shoulders_x - self.previous_shoulders_position

        # Store current position for next frame
        self.previous_shoulders_position = shoulders_x

        # Handle horizontal movement
        self.handle_horizontal_movement(shoulders_x, delta_x)

        # Uncomment to also implement vertical movement if needed
        # nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        # self.handle_vertical_movement(nose.y)

    def handle_horizontal_movement(self, current_position, delta_x):
        """Handle horizontal movement based on shoulder position

        Args:
            current_position: Current horizontal position of shoulders
            delta_x: Change in position since last frame
        """
        # If we're close to the reference position, reset
        if (
            abs(current_position - self.reference_shoulders_position)
            < self.horizontal_movement_threshold
        ):
            self.reference_shoulders_position = current_position

            if self.horizontal_key is not None:
                simulate_key_release(self.horizontal_key)
                self.horizontal_key = None
                print("Movement stopped")
            return

        # Check if movement exceeds sensitivity threshold
        if (
            abs(current_position - self.reference_shoulders_position)
            < self.horizontal_sensitivity
        ):
            return

        # Determine direction (left/right)
        new_key = None
        if current_position < self.reference_shoulders_position:
            new_key = KEY_MAPPINGS["left"]
            direction = "left"
        else:
            new_key = KEY_MAPPINGS["right"]
            direction = "right"

        # If direction changed, release old key
        if self.horizontal_key is not None and new_key != self.horizontal_key:
            simulate_key_release(self.horizontal_key)

        # Press the new key
        if self.horizontal_key != new_key:
            self.horizontal_key = new_key
            simulate_key_press(self.horizontal_key)
            print(f"Moving {direction}")

    def start(self):
        """Start the pose-based gesture control"""
        print("Starting pose gesture control...")

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Pose tracking activated. Press ESC to exit.")
        print("Movement controls: Lean left/right to move your character")

        while self.cap.isOpened():
            frame = capture_frame(self.cap)
            if frame is None:
                break

            # Process the frame with MediaPipe Pose
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get right wrist position
                left_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX]
                right_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX]

                # Get knee position
                left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]

                # Update face position from nose landmark
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                self.face_position = (nose.x, nose.y)

                # Draw landmarks for visualization
                draw_landmarks(frame, landmarks, self.mp_pose)

                # Process movement using shoulder position
                self.process_movement(landmarks)

                # Check if wrist is in trigger zone, using face position
                now = time.time()
                if point_in_trigger_zone(
                    left_hand.x, left_hand.y, self.trigger_zone, self.face_position
                ):
                    if now - self.last_action_time > self.cooldown:
                        self.trigger_action("punch")
                if point_in_trigger_zone(
                    right_hand.x, right_hand.y, self.trigger_zone, self.face_position
                ):
                    if now - self.last_action_time > self.cooldown:
                        self.trigger_action("block")
                if point_in_trigger_zone(
                    left_knee.x, left_knee.y, self.trigger_zone, self.face_position
                ):
                    if now - self.last_action_time > self.cooldown:
                        self.trigger_action("kick2")
                if point_in_trigger_zone(
                    right_knee.x, right_knee.y, self.trigger_zone, self.face_position
                ):
                    if now - self.last_action_time > self.cooldown:
                        self.trigger_action("kick")

            # Draw the trigger zone relative to face position
            draw_trigger_zone(frame, self.trigger_zone, self.face_position)

            cv2.imshow("Pose Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

        # Clean up
        if self.horizontal_key is not None:
            simulate_key_release(self.horizontal_key)

        self.cap.release()
        cv2.destroyAllWindows()
        print("Pose gesture control terminated.")

    def stop(self):
        """Stop the gesture control and release resources"""
        if self.horizontal_key is not None:
            simulate_key_release(self.horizontal_key)

        if self.cap and self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()


def start_pose_gaming():
    """Main function to start pose-based game control"""
    controller = PoseGestureController()
    try:
        controller.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        controller.stop()


if __name__ == "__main__":
    start_pose_gaming()
