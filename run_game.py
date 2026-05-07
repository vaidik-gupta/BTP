import argparse
import time
from collections import deque

import numpy as np
import tensorflow as tf
from pylsl import StreamInlet, resolve_byprop

from eeg_game import EEGConeGame

FS = 250  # EEG Sampling Frequency (Hz)
TIME_WINDOW = 1.0  # Time window for each action (seconds)
SAMPLES_PER_ACTION = int(FS * TIME_WINDOW)
ACTION_INTERVAL_SEC = 0.5


def resolve_eeg_stream(timeout=10):
    print("Looking for EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=timeout)
    if not streams:
        raise RuntimeError("No EEG stream found. Make sure your device is streaming and try again.")
    inlet = StreamInlet(streams[0])
    print("Connected to EEG stream")
    return inlet


def update_eeg_buffer(inlet, eeg_buffer):
    while True:
        sample, timestamp = inlet.pull_sample(timeout=0.0)
        if sample is None:
            break
        eeg_buffer.append(sample[:8])
    return eeg_buffer


def prepare_eeg_input(buffer, time_progress):
    window = np.array(buffer)[-SAMPLES_PER_ACTION:]
    window = window.T.astype(np.float32)

    time_channel = np.full((1, SAMPLES_PER_ACTION), time_progress, dtype=np.float32)
    window = np.concatenate([window, time_channel], axis=0)
    window = np.expand_dims(window, axis=-1)
    return np.expand_dims(window, axis=0)


def run_policy_test(policy_model_path, episodes=5, time_minutes=0.75, speed=20):
    policy_model = tf.keras.models.load_model(policy_model_path)
    inlet = resolve_eeg_stream(timeout=10)
    game = EEGConeGame(time_minutes=time_minutes, speed=speed)

    for episode in range(episodes):
        print(f"\n=== Test episode {episode + 1}/{episodes} ===")
        game.reset()
        done = False
        total_score = 0.0
        eeg_buffer = deque(maxlen=SAMPLES_PER_ACTION)
        episode_start_time = time.time()
        next_decision_time = episode_start_time

        while not done:
            update_eeg_buffer(inlet, eeg_buffer)
            now = time.time()
            action = 1

            if len(eeg_buffer) == SAMPLES_PER_ACTION and now >= next_decision_time:
                time_progress = (now - episode_start_time) / (2.0 * 60.0)
                input_state = prepare_eeg_input(eeg_buffer, time_progress)
                probs = policy_model(input_state, training=False).numpy()[0]
                probs = np.clip(probs, 1e-10, 1.0)
                probs = probs / np.sum(probs)
                action = int(np.argmax(probs))
                next_decision_time = now + ACTION_INTERVAL_SEC

            _, reward, done = game.step(action)
            total_score += reward
            game.score = total_score
            game.render()

        print(f"Episode {episode + 1} finished. Total score: {total_score:.1f}")

    print("\nPolicy test complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run EEG policy model test episodes.")
    parser.add_argument("policy_model", help="Path to the saved policy model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes to run")
    parser.add_argument("--time-minutes", type=float, default=0.75, help="Length of each game episode in minutes")
    parser.add_argument("--speed", type=int, default=20, help="Game speed for each episode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_policy_test(
        policy_model_path=args.policy_model,
        episodes=args.episodes,
        time_minutes=args.time_minutes,
        speed=args.speed,
    )
