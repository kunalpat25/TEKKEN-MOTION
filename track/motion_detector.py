import numpy as np
import cv2
import time

import imutils
from imutils.video import FileVideoStream, VideoStream
from pynput.keyboard import Key

from track.utils import (
    CAMERA_SOURCE,
    SETUP_COUNTDOWN,
    POSE_COUNTDOWN,
    TEXT_POSITION,
    NOTIFICATION_COLOR,
    load_configuration,
)
from track.utils import (
    capture_frame_cv2,
    capture_frame_imutils,
    calculate_center_point,
    render_rectangle,
)
from core.settings import KEY_MAPPINGS

# Default configuration values
DEFAULT_FACE_TRACKING_REGION = (245, 87, 107, 136)
DEFAULT_GESTURE_DETECTION_REGIONS = [
    (460, 102, 107, 92),
    (319, 270, 80, 81),
]


def get_configuration():
    """Load configuration from file or return default if file not found"""
    face_tracking_region = DEFAULT_FACE_TRACKING_REGION
    gesture_detection_regions = DEFAULT_GESTURE_DETECTION_REGIONS
    tracking_center = calculate_center_point(face_tracking_region)
    gesture_regions_count = len(gesture_detection_regions)

    loaded_config = load_configuration()
    if loaded_config:
        (
            face_tracking_region,
            tracking_center,
            gesture_regions_count,
            gesture_detection_regions,
        ) = loaded_config

    return [
        face_tracking_region,
        tracking_center,
        gesture_regions_count,
        gesture_detection_regions,
    ]


(
    face_tracking_region,
    tracking_center,
    gesture_regions_count,
    gesture_detection_regions,
) = get_configuration()

DETECTION_THRESHOLD = 1000


class MotionRegion:
    def __init__(
        self, tracking_region, detection_region=None, sensitivity=DETECTION_THRESHOLD
    ):
        self.sensitivity = sensitivity
        self.last_detection = time.time()
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        if detection_region:
            self.detection_area = detection_region
        else:
            self.configure_region(tracking_region)

    def configure_region(self, tracking_region):
        video_stream = cv2.VideoCapture(CAMERA_SOURCE)
        time.sleep(0.2)
        configuration_time = POSE_COUNTDOWN
        start_time = time.time()

        while True:
            frame = capture_frame_cv2(video_stream)
            if frame is None:
                break

            elapsed_time = time.time() - start_time
            if elapsed_time > configuration_time:
                break

            cv2.putText(
                frame,
                str(int(configuration_time - elapsed_time) + 1),
                TEXT_POSITION,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                NOTIFICATION_COLOR,
                4,
            )
            render_rectangle(True, tracking_region, frame)
            cv2.imshow("Region Configuration", frame)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        cv2.putText(
            frame,
            "Select Motion Detection Region",
            TEXT_POSITION,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            NOTIFICATION_COLOR,
            2,
        )
        self.detection_area = cv2.selectROI(frame, False)
        cv2.destroyAllWindows()
        video_stream.release()

    def analyze_frame(self, frame):
        try:
            x, y, w, h = self.detection_area
            region = frame[y : y + h, x : x + w]

            foreground_mask = self.motion_detector.apply(region)

            foreground_mask = cv2.morphologyEx(
                foreground_mask, cv2.MORPH_ERODE, self.morph_kernel, iterations=2
            )
        except:
            return False

        motion_intensity = np.sum(foreground_mask == 255)

        cv2.putText(
            frame,
            str(motion_intensity),
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            NOTIFICATION_COLOR,
            2,
        )
        render_rectangle(True, self.detection_area, frame)

        if time.time() - self.last_detection < 0.05:
            render_rectangle(True, self.detection_area, frame)

        if (
            motion_intensity > self.sensitivity
            and time.time() - self.last_detection > 0.2
        ):
            render_rectangle(True, self.detection_area, frame)
            self.last_detection = time.time()
            return True

        return False


class GestureController:
    def __init__(self, face_region=None, regions_count=0, training_mode=False):
        self.training_mode = training_mode

        if not training_mode:
            self.load_configuration()
        else:
            self.regions_count = regions_count
            self.gesture_regions = []
            self.face_region = face_region
            self.face_center = calculate_center_point(self.face_region)

        self.initialize_detection_regions()
        self.input_controller = InputController(self.face_center)

    def load_configuration(self):
        config = get_configuration()
        self.face_region, self.face_center, self.regions_count, self.gesture_regions = (
            config
        )

    def initialize_detection_regions(self):
        self.detectors = []
        self.relative_positions = []

        for i in range(self.regions_count):
            if not self.training_mode:
                region = self.gesture_regions[i]
                detector = MotionRegion(self.face_region, region)
            else:
                detector = MotionRegion(self.face_region)
                self.gesture_regions.append(detector.detection_area)

            self.detectors.append(detector)
            self.relative_positions.append(
                self.calculate_relative_position(detector.detection_area)
            )

        if self.training_mode:
            print(
                f"face_tracking_region = {self.face_region} \ngesture_detection_regions = {self.gesture_regions}"
            )

    def calculate_relative_position(self, region):
        center_x, center_y = self.face_center
        x, y, w, h = region
        dx, dy = x - center_x, y - center_y
        return dx, dy, w, h

    def update_region_position(self, relative_position):
        current_x, current_y = self.current_face_center
        dx, dy, w, h = relative_position
        x = dx + current_x
        y = dy + current_y
        return x, y, w, h

    def process_frame(self, frame, current_face_center):
        self.current_face_center = current_face_center

        for i in range(self.regions_count):
            detector = self.detectors[i]
            relative_pos = self.relative_positions[i]

            detector.detection_area = self.update_region_position(relative_pos)
            gesture_detected = detector.analyze_frame(frame)

            if gesture_detected:
                self.input_controller.execute_action(i)

        self.input_controller.process_movement(current_face_center)


from core.input import simulate_key_press, simulate_key_release


class InputController:
    def __init__(self, initial_position):
        self.face_center = initial_position
        self.movement_threshold = 2
        self.previous_position = initial_position
        self.neutral_position = initial_position
        self.horizontal_key = None
        self.vertical_key = None
        self.reference_height = initial_position[1]
        self.vertical_sensitivity = 40
        self.horizontal_sensitivity = 10
        # Use key mappings from central settings
        self.action_mapping = {
            0: KEY_MAPPINGS["punch"],  # punch
            1: KEY_MAPPINGS["kick"],  # kick
            2: KEY_MAPPINGS["kick2"],  # power move
        }

    def execute_action(self, action_index):
        if self.horizontal_key is not None:
            return

        key_code = self.get_key_for_action(action_index)
        self.trigger_key_press(key_code)

    def get_key_for_action(self, action_index):
        return self.action_mapping.get(
            action_index, KEY_MAPPINGS["punch"]
        )  # Default to punch

    def trigger_key_press(self, key_code, continuous=False):
        simulate_key_press(key_code)

        if not continuous:
            time.sleep(0.07)
            simulate_key_release(key_code)

        time.sleep(0.01)

    def process_movement(self, current_position):
        self.current_position = current_position

        delta_x = self.current_position[0] - self.previous_position[0]
        delta_y = self.current_position[1] - self.reference_height

        if (
            abs(self.current_position[0] - self.neutral_position[0])
            > self.horizontal_sensitivity
        ):
            intensity = 1
            if delta_x > 0:
                intensity = 2

            for _ in range(intensity):
                self.handle_horizontal_movement(delta_x)

        if abs(delta_y) > self.vertical_sensitivity:
            self.handle_vertical_movement(delta_y)

        self.previous_position = current_position

    def handle_horizontal_movement(self, delta):
        if abs(delta) < self.movement_threshold:
            self.neutral_position = self.current_position

            if self.horizontal_key is not None:
                simulate_key_release(self.horizontal_key)
                self.horizontal_key = None
            return

        if delta < 0:
            self.horizontal_key = KEY_MAPPINGS["left"]
        else:
            self.horizontal_key = KEY_MAPPINGS["right"]

        self.trigger_key_press(self.horizontal_key, True)

    def handle_vertical_movement(self, delta):
        if delta < 50 and self.vertical_key == KEY_MAPPINGS["down"]:
            simulate_key_release(self.vertical_key)
            self.vertical_key = None

        continuous = False
        if delta < 0 and delta < -25:
            self.vertical_key = KEY_MAPPINGS["up"]  # Jump
        elif delta > 50:
            self.vertical_key = KEY_MAPPINGS["down"]  # Crouch
            continuous = True
        else:
            return

        self.trigger_key_press(self.vertical_key, continuous)
