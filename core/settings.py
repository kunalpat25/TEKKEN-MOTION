"""
Common settings file for Gesture Gaming Tekken
Contains shared configuration used by both pose and tracking variants
"""

import json
import os
from pynput.keyboard import Key


# Key mappings for game controls
DEFAULT_KEY_MAPPINGS = {
    # Movement keys
    "up": Key.up,  # Jump
    "down": Key.down,  # Crouch
    "left": Key.left,  # Move left
    "right": Key.right,  # Move right
    # Action keys
    "punch": "a",  # Punch action
    "kick": "x",  # Kick action
    "kick2": "z",  # Power move action
    # Additional keys that might be useful
    "block": "s",  # Block action
}

# Path to the key mappings configuration file
KEY_CONFIG_FILE_PATH = "key_config.json"


def save_key_mappings(key_mappings=None):
    """Save key mappings to configuration file

    For special keys like Key.up, we save them as string representations
    that can be converted back when loaded
    """
    if key_mappings is None:
        key_mappings = DEFAULT_KEY_MAPPINGS

    # Convert pynput.keyboard.Key objects to string representations
    serializable_mappings = {}
    for key, value in key_mappings.items():
        if isinstance(value, Key):
            # Store special keys with a prefix so we know to convert them back
            serializable_mappings[key] = f"__special_key__{value.name}"
        else:
            serializable_mappings[key] = value

    with open(KEY_CONFIG_FILE_PATH, "w") as config_file:
        json.dump(serializable_mappings, config_file, indent=4)

    print(f"Key mappings saved to {KEY_CONFIG_FILE_PATH}")


def load_key_mappings():
    """Load key mappings from configuration file

    If file doesn't exist, create it with default mappings
    """
    if not os.path.exists(KEY_CONFIG_FILE_PATH):
        print(f"Key mappings file not found. Creating with defaults.")
        save_key_mappings(DEFAULT_KEY_MAPPINGS)
        return DEFAULT_KEY_MAPPINGS

    try:
        with open(KEY_CONFIG_FILE_PATH, "r") as config_file:
            serialized_mappings = json.load(config_file)

        # Convert string representations back to Key objects
        key_mappings = {}
        for key, value in serialized_mappings.items():
            if isinstance(value, str) and value.startswith("__special_key__"):
                # Convert back to pynput.keyboard.Key object
                key_name = value.replace("__special_key__", "")
                key_mappings[key] = getattr(Key, key_name)
            else:
                key_mappings[key] = value

        return key_mappings
    except Exception as e:
        print(f"Error loading key mappings: {e}")
        print("Using default key mappings instead")
        return DEFAULT_KEY_MAPPINGS


# Create global instance of key mappings that can be imported
KEY_MAPPINGS = load_key_mappings()


if __name__ == "__main__":
    # When run directly, display current key mappings
    print("Current key mappings:")
    for action, key in KEY_MAPPINGS.items():
        print(f"{action}: {key}")
