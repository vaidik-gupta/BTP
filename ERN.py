import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylsl import StreamInlet, resolve_byprop

N_DATASET = 2

FS = 250                  # EEG sampling rate (change to your device rate)
WINDOW_SEC1 = 3            # record 3 second windows as requested
WINDOW_SEC2 = 1.5
CHANNELS = 8

TRIALS = 100               # adjust trial count as needed

DIRECTIONS = ["Up", "Down", "Left", "Right"]


def collect_window(inlet, duration_sec=WINDOW_SEC1):
    buffer = []
    while len(buffer) < int(FS * duration_sec):
        sample, timestamp = inlet.pull_sample()
        buffer.append(sample[:CHANNELS])
    return np.array(buffer)


def show_direction_image(direction):
    filename = f"ERN_images/{direction}.png"

    try:
        image_data = mpimg.imread(filename)
    except Exception as e:
        print(f"Could not open image {filename}: {e}")
        return None

    has_gui = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland")
    if not has_gui:
        print(f"Headless environment detected; skipping image display: {filename}")
        return None

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(image_data)
    plt.show(block = False)
    plt.pause(0.75)  # brief pause to ensure image is rendered
    plt.close(fig)


def run_experiment():
    print("Looking for EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=10)
    if not streams:
        raise RuntimeError("No EEG stream found. Make sure your device is streaming and try again.")

    inlet = StreamInlet(streams[0])
    print("Connected to EEG stream")

    dataset_thought = []   # dataset 1: EEG for direction you were thinking (final label is the actual direction)
    labels_thought = []

    dataset_feedback = []  # dataset 2: EEG after image shown (label is correctness 1/-1)
    labels_feedback = []

    print("Starting experiment in 5 seconds")
    time.sleep(5)

    for trial in range(1, TRIALS + 1):
        print(f"\n=== Trial {trial}/{TRIALS} ===")

        # 1) Subject thinks of a direction
        chosen_dir = random.choice(DIRECTIONS)
        print(chosen_dir)
        time.sleep(0.2)

        thought_window = collect_window(inlet, WINDOW_SEC1)

        # 2) Show random direction image (guess step)
        guess_dir = random.choice(DIRECTIONS)
        show_direction_image(guess_dir)

        # Keep the image on screen for at least 1 second before recording.

        print("Recording EEG after shown direction")
        feedback_window = collect_window(inlet, duration_sec=WINDOW_SEC2)


        correct = 1 if guess_dir == chosen_dir else -1

        dataset_thought.append(thought_window)
        labels_thought.append(chosen_dir)

        dataset_feedback.append(feedback_window)
        labels_feedback.append(correct)

        print(f"trial {trial}: thought={chosen_dir}, shown={guess_dir}, correct={correct}")
        time.sleep(1)

    # Convert to numpy and save
    X1 = np.array(dataset_thought)
    y1 = np.array(labels_thought)

    X2 = np.array(dataset_feedback)
    y2 = np.array(labels_feedback)

    np.save(f"direction_waves_X{N_DATASET}.npy", X1)
    np.save(f"direction_waves_y{N_DATASET}.npy", y1)
    np.save(f"ERN_X{N_DATASET}.npy", X2)
    np.save(f"ERN_y{N_DATASET}.npy", y2)

    print("Saved datasets:")
    print(" - direction_waves_X.npy", X1.shape)
    print(" - direction_waves_y.npy", y1.shape)
    print(" - ERN_X.npy", X2.shape)
    print(" - ERN_y.npy", y2.shape)


if __name__ == "__main__":
    run_experiment()


