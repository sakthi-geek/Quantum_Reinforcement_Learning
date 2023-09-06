import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from collections import defaultdict
import os


eps = np.finfo(np.float32).eps.item() # Smallest number such that 1.0 + eps != 1.0

class ActorCriticClassical:
    def __init__(self, gamma=0.99, n_episodes=3000, max_steps_per_episode=10000, learning_rate=0.01):
        # Configuration parameters
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = learning_rate

        self.env = gym.make("CartPole-v1")

        #----- Model ---------------------------------------
        num_inputs = 4
        num_actions = 2
        num_hidden = 128

        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        action = layers.Dense(num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        self.model = keras.Model(inputs=inputs, outputs=[action, critic])
        print(self.model.summary())
        self.input_shape = self.model.input_shape
        self.output_shape = self.model.output_shape
        self.trainable_params = self.model.count_params()
        print(self.input_shape, self.output_shape, self.trainable_params)

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.huber_loss = keras.losses.Huber()

        # Metrics ---------------------------
        self.episode_reward_history = []
        self.episode_length_history = []

    def train(self):
        episode_count = 0
        running_reward = 0
        # N = 1  # You can adjust this to capture metrics every N episodes

        for _ in range(self.n_episodes):
            state = self.env.reset()
            action_probs_history = []
            critic_value_history = []
            rewards_history = []
            episode_reward = 0
            episode_length = 0

            with tf.GradientTape() as tape:
                for timestep in range(1, self.max_steps_per_episode):
                    state = tf.convert_to_tensor(state)
                    state = tf.expand_dims(state, 0)
                    action_probs, critic_value = self.model(state)
                    critic_value_history.append(critic_value[0, 0])
                    action = np.random.choice(self.env.action_space.n, p=np.squeeze(action_probs))
                    action_probs_history.append(tf.math.log(action_probs[0, action]))
                    state, reward, done, _ = self.env.step(action)

                    rewards_history.append(reward)
                    episode_reward += reward
                    episode_length += 1

                    if done:
                        break

                running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
                self.episode_reward_history.append(episode_reward)
                self.episode_length_history.append(episode_length)

                returns = []
                discounted_sum = 0
                for r in rewards_history[::-1]:
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
                returns = returns.tolist()

                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in zip(action_probs_history, critic_value_history, returns):
                    diff = ret - value
                    actor_losses.append(-log_prob * diff)
                    critic_losses.append(self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            episode_count += 1
            if episode_count % 10 == 0:
                avg_rewards = np.mean(self.episode_reward_history[-10:])
                print("Episode {}/{}, average last 10 rewards {}".format(episode_count, self.n_episodes, avg_rewards))

                if avg_rewards >= 500.0:
                    break

        # # Storing rewards stats - GRANULARITY: episodes
        self.episode_count = len(self.episode_reward_history)
        self.average_timesteps_per_episode = np.mean(self.episode_length_history)
        self.episode_rewards_min = np.min(self.episode_reward_history)
        self.episode_rewards_max = np.max(self.episode_reward_history)
        self.episode_rewards_mean = np.mean(self.episode_reward_history)
        self.episode_rewards_median = np.median(self.episode_reward_history)
        self.episode_rewards_std = np.std(self.episode_reward_history)
        self.episode_rewards_iqr = np.subtract(*np.percentile(self.episode_reward_history, [75, 25]))
        self.episode_rewards_q1 = np.percentile(self.episode_reward_history, 25)
        self.episode_rewards_q3 = np.percentile(self.episode_reward_history, 75)


        self.plot_rewards()

    def save_model(self, path):
        """Save the model to the specified path."""
        self.model.save(path)

    def load_model(self, path):
        """Load the model from the specified path."""
        self.model = tf.keras.models.load_model(path)

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_reward_history)
        plt.xlabel('Epsiode')
        plt.ylabel('Collected rewards')
        plt.show()

    def play(self, episodes=5, render=False):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                if render:
                    self.env.render()
                # state = tf.convert_to_tensor(state)
                # state = tf.expand_dims(state, 0)
                action_probs, _ = self.model(state[None, :])
                action = np.argmax(action_probs[0])
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        if render:
            self.env.close()


if __name__ == "__main__":
    actor_critic_agent = ActorCriticClassical()
    actor_critic_agent.train()
    actor_critic_agent.play(render=True)
