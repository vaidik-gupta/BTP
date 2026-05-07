import os
import re
import time
from collections import deque

import numpy as np
import tensorflow as tf
from pylsl import StreamInlet, resolve_byprop

from eeg_game import EEGConeGame
from PPO import PPOAgentTF
from eeg_net import EEGNet

FS = 250  # EEG Sampling Frequency (Hz)
TIME_WINDOW = 1.0  # Time window for each action (seconds)
SAMPLES_PER_ACTION = int(FS * TIME_WINDOW)
ACTION_INTERVAL_SEC = 0.5
EPISODES = 100
GAMMA = 0.9
PPO_UPDATE_EPOCHS = 4
MOVEMENT_PENALTY = 0.3
PAUSE_AFTER_EPISODES = 10

PLAYER = "VAIDIK"
TRIAL = 2
MODEL_DIR = "ConeGameModels"

# EEG channels + 1 for time
CHANS = 9


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
    window = window.T.astype(np.float32)  # (8, 250)
    
    # Add time as an extra channel
    time_channel = np.full((1, SAMPLES_PER_ACTION), time_progress, dtype=np.float32)
    window = np.concatenate([window, time_channel], axis=0)  # (9, 250)
    
    window = np.expand_dims(window, axis=-1)  # (9, 250, 1)
    return np.expand_dims(window, axis=0)  # (1, 9, 250, 1)


def discounted_returns(rewards, gamma=0.99):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_total = 0.0
    for idx in reversed(range(len(rewards))):
        running_total = rewards[idx] + gamma * running_total
        returns[idx] = running_total
    return returns


def train_from_episode(agent, states, actions, old_log_probs, returns, advantages, epochs=4):
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
    returns = tf.convert_to_tensor(returns, dtype=tf.float32)
    advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

    for _ in range(epochs):
        loss, policy_loss, value_loss = agent.train_step(
            states, actions, old_log_probs, returns, advantages
        )
    return loss, policy_loss, value_loss


def save_model(policy_model, value_model, episode):
    os.makedirs(MODEL_DIR, exist_ok=True)
    filename = f"{PLAYER}_trial{TRIAL}_episode{episode + 1:02d}_policy.keras"
    filename2 = f"{PLAYER}_trial{TRIAL}_episode{episode + 1:02d}_value.keras"
    save_path = os.path.join(MODEL_DIR, filename)
    save_path2 = os.path.join(MODEL_DIR, filename2)
    policy_model.save(save_path)
    print(f"Saved policy model: {save_path}")
    value_model.save(save_path2)
    print(f"Saved value model: {save_path2}")


def main():
    game = EEGConeGame(time_minutes=0.75, speed=20)
    inlet = resolve_eeg_stream(timeout=10)

    policy_model = EEGNet(nb_classes=3, Chans=CHANS, Samples=SAMPLES_PER_ACTION)
    value_model = EEGNet(nb_classes=1, Chans=CHANS, Samples=SAMPLES_PER_ACTION, value=True)
    agent = PPOAgentTF(policy_model, value_model, lr=3e-4, epsilon=0.2)

    for episode in range(EPISODES):

        if episode % 10 == 0 and episode > 0:
            save_model(policy_model, value_model, episode)

            input("Press Enter to continue...")
            


        print("Next episode starting in 5 sec")

        time.sleep(5)

        state = game.reset()
        done = False
        total_score = 0.0
        eeg_buffer = deque(maxlen=SAMPLES_PER_ACTION)
        episode_start_time = time.time()
        next_decision_time = episode_start_time
        last_action = None
        action_reward = 0.0

        states = []
        actions = []
        old_log_probs = []
        values = []
        rewards = []

        while not done:
            update_eeg_buffer(inlet, eeg_buffer)
            now = time.time()
            decision_step = len(eeg_buffer) == SAMPLES_PER_ACTION and now >= next_decision_time

            if decision_step:
                time_progress = (now - episode_start_time) / (2.0 * 60.0)  # Normalize to [0, 1]
                input_state = prepare_eeg_input(eeg_buffer, time_progress)
                probs = policy_model(input_state, training=False).numpy()[0]
                action = np.random.choice(3, p=probs)
                action_log_prob = np.log(np.clip(probs[action], 1e-10, 1.0))
                value = value_model(input_state, training=False).numpy()[0, 0]

                if last_action is not None:
                    rewards.append(action_reward)

                states.append(input_state[0])
                actions.append(action)
                old_log_probs.append(action_log_prob)
                values.append(value)
                action_reward = 0.0
                last_action = action
                next_decision_time = now + ACTION_INTERVAL_SEC
                frame_action = action
            else:
                frame_action = 1

            _, reward, done = game.step(frame_action)
            # Apply movement penalty for left/right actions
            if frame_action in [0, 2]:
                reward -= MOVEMENT_PENALTY
            total_score += reward
            action_reward += reward
            game.score = total_score
            game.render()

        if last_action is not None:
            rewards.append(action_reward)

        if len(rewards) > 0:
            returns = discounted_returns(rewards, gamma=GAMMA)
            advantages = returns - np.array(values, dtype=np.float32)
            loss, policy_loss, value_loss = train_from_episode(
                agent,
                np.array(states),
                np.array(actions),
                np.array(old_log_probs),
                returns,
                advantages,
                epochs=PPO_UPDATE_EPOCHS,
            )
            print(
                f"Episode {episode + 1}/{EPISODES} complete. Score: {total_score:.1f}, "
                f"loss={loss:.4f}, policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}"
            )
        else:
            print(f"Episode {episode + 1}/{EPISODES} complete. Score: {total_score:.1f}, no training samples collected.")





    print("Training complete.")


def main2():
    ## dummy run without model to test game loop
    game = EEGConeGame(time_minutes=0.2, speed=20)
    for episode in range(1):
        state = game.reset()
        done = False
        total_score = 0.0

        while not done:
            action = np.random.choice(3, p=[0.02, 0.96, 0.02])
            _, reward, done = game.step(action)
            total_score += reward
            game.score = total_score
            game.render()

        print(f"Episode {episode + 1} complete. Score: {total_score:.1f}")


if __name__ == "__main__":
    # main2()
    main()
