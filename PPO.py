import tensorflow as tf
import numpy as np

class PPOAgentTF:
    def __init__(self, policy_model, value_model, lr=3e-4, epsilon=0.2):
        self.policy = policy_model
        self.critic = value_model
        self.epsilon = epsilon
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def get_log_probs(self, model, states, actions):
        probs = model(states)

        indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        action_probs = tf.gather_nd(probs, indices)

        # safer log
        action_probs = tf.clip_by_value(action_probs, 1e-10, 1.0)
        return tf.math.log(action_probs)

    @tf.function
    def train_step(self, states, actions, old_log_probs, returns, advantages):

        with tf.GradientTape() as tape:
            # --- Policy ---
            curr_log_probs = self.get_log_probs(self.policy, states, actions)
            ratio = tf.exp(curr_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages

            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # --- Value ---
            state_values = tf.squeeze(self.critic(states), axis=-1)
            value_loss = tf.reduce_mean(tf.square(returns - state_values))

            # --- Total ---
            total_loss = policy_loss + 0.5 * value_loss

        trainable_vars = self.policy.trainable_variables + self.critic.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Gradient clipping (important for PPO stability)
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return total_loss, policy_loss, value_loss