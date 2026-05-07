import random
import time
import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_byprop

# ----------------------------
# Parameters
# ----------------------------

FS = 250                  # EEG sampling rate (change to your device rate)
WINDOW_SEC = 2            # wave packet size
WINDOW_SAMPLES = FS * WINDOW_SEC
CHANNELS = 8

TRIALS = 100            # total trials

# CUES = [
#     "wink",
#     "fist_right",
#     "fist_left",
#     "tongue"
# ]

CUES = [
   "right_curl",
    "left_curl"
]

REST_LABEL = "rest"

dataset = []
labels = []

# ----------------------------
# Connect to LSL
# ----------------------------

print("Looking for EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=5)

if not streams:
    print("No EEG stream found. Make sure your device is streaming and try again.")
    exit()

inlet = StreamInlet(streams[0])

print("Connected to EEG stream")

# ----------------------------
# Helper function to collect window
# ----------------------------

def collect_window():

    data = []

    while len(data) < WINDOW_SAMPLES:

        sample, timestamp = inlet.pull_sample()

        data.append(sample[:CHANNELS])

    return np.array(data)


# ----------------------------
# Trial Loop
# ----------------------------

print("Starting experiment in 5 seconds")
time.sleep(5)

for i in range(TRIALS):

    # ------------------------
    # REST PERIOD
    # ------------------------

    print("\nREST")

    rest_data = collect_window()

    dataset.append(rest_data)
    labels.append(REST_LABEL)

    time.sleep(1)

    # ------------------------
    # RANDOM CUE
    # ------------------------

    cue = random.choice(CUES)

    print("CUE:", cue)

    time.sleep(0.5)

    trial_data = collect_window()

    dataset.append(trial_data)
    labels.append(cue)

    time.sleep(1)

# ----------------------------
# Convert dataset
# ----------------------------

X = np.array(dataset)      # shape: (trials, samples, channels)
y = np.array(labels)

print("Dataset shape:", X.shape)

# ----------------------------
# Save dataset
# ----------------------------

np.save("eeg_dataset_X_devansh2.npy", X)
np.save("eeg_dataset_y_devansh2.npy", y)

print("Dataset saved!")