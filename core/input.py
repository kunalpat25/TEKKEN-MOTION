import time
import platform
from pynput.keyboard import Key, Controller

keyboard_controller = Controller()
OS_PLATFORM = platform.system()


def simulate_key_press(key: Key):
    keyboard_controller.press(key)


def simulate_key_release(key: Key):
    keyboard_controller.release(key)


if __name__ == "__main__":
    print("Testing keyboard simulation...")
    print("Press and release 'W' key in 3 seconds...")
    time.sleep(3)
    simulate_key_press("w")
    time.sleep(1)
    simulate_key_release("w")
    print("Test completed")
