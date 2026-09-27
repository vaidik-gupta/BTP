import numpy as np
import tensorflow as tf

from eeg_bci.models.ppo import PPOAgentTF


class SimpleEnv:
    """A simple deterministic contextual bandit with two actions."""

    def __init__(self, state_dim=4, max_steps=8):
        self.state_dim = state_dim
        self.max_steps = max_steps
        self.step_count = 0
        self.state = None

    def reset(self):
        self.step_count = 0
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        return self.state

    def step(self, action):
        assert action in (0, 1)
        target_action = 1 if (self.state[0] + self.state[1] - self.state[2]) > 0 else 0
        reward = 1.0 if action == target_action else 0.0
        self.step_count += 1
        done = self.step_count >= self.max_steps
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        return self.state, reward, done


def discounted_returns(rewards, gamma=0.99):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_total = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running_total = rewards[idx] + gamma * running_total
        returns[idx] = running_total
    return returns


def make_models(state_dim=4, n_actions=2):
    policy = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(state_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(n_actions, activation="softmax"),
        ]
    )

    critic = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(state_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    return policy, critic


def run_training(episodes=300, print_every=25):
    env = SimpleEnv()
    policy_model, value_model = make_models()
    agent = PPOAgentTF(policy_model, value_model, lr=2e-3, epsilon=0.2)

    episode_rewards = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        states = []
        actions = []
        old_log_probs = []
        rewards = []

        while not done:
            state_batch = np.expand_dims(state, axis=0)
            probs = policy_model(state_batch, training=False).numpy()[0]
            action = np.random.choice(2, p=probs)
            log_prob = np.log(np.clip(probs[action], 1e-10, 1.0))

            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        returns = discounted_returns(rewards)
        values = value_model(np.array(states, dtype=np.float32), training=False).numpy().flatten()
        advantages = returns - values

        agent.train_step(
            tf.convert_to_tensor(np.array(states, dtype=np.float32), dtype=tf.float32),
            tf.convert_to_tensor(np.array(actions, dtype=np.int32), dtype=tf.int32),
            tf.convert_to_tensor(np.array(old_log_probs, dtype=np.float32), dtype=tf.float32),
            tf.convert_to_tensor(returns, dtype=tf.float32),
            tf.convert_to_tensor(advantages, dtype=tf.float32),
        )

        episode_rewards.append(sum(rewards))

        if ep % print_every == 0:
            average_reward = np.mean(episode_rewards[-print_every:])
            print(f"Episode {ep}/{episodes}  avg reward last {print_every}: {average_reward:.3f}")

    overall_avg = np.mean(episode_rewards[-print_every:])
    print(f"Training finished. Final average reward over last {print_every} episodes: {overall_avg:.3f}")


if __name__ == "__main__":
    run_training()
