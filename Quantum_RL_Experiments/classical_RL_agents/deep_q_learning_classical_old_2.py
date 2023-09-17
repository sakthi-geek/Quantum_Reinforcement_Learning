
# Importing required libraries
import gym
import numpy as np
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay


class DeepQLearningClassical:
    def __init__(self, env_name='CartPole-v1', seed=39, n_inputs=4, n_hidden=[32], n_actions=2,
                 gamma=0.99, n_episodes=3000, batch_size=16, learning_rate=0.001):

        # Environment -----------------------
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.seed = seed
        self.env.seed(self.seed)

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        random.seed(self.seed)

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.learning_rate = learning_rate

        # Optimizer with Learning Rate Scheduler
        self.lr_schedule = ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,  # decay the learning rate after every 10000 steps
            decay_rate=0.99)  # decay rate factor

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)  # amsgrad=True is default

        # I Initialize Q-Network and Target Network
        self.q_network = self.build_network()
        self.target_network = self.build_network()
        self.hard_update_target_network()

        tf.keras.utils.plot_model(self.q_network, show_shapes=True, dpi=70)
        tf.keras.utils.plot_model(self.target_network, show_shapes=True, dpi=70)

        self.input_shape = self.q_network.input_shape
        self.output_shape = self.q_network.output_shape
        self.trainable_params = self.q_network.count_params()

        # Initialize replay memory and other variables
        self.max_memory_length = 10000
        self.replay_memory = deque(maxlen=self.max_memory_length)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.decay_epsilon = 0.995 # Decay rate of epsilon greedy parameter - range - 0 to 1 - 1 means decay is not happening - 0 means decay is happening very fast
        self.steps_per_update = 10
        self.steps_per_target_update = 50


        self.config_params = None
        # print all parameters
        print("----------------------")
        print("Environment: ", self.env_name)
        print("Seed: ", self.seed)
        print("Gamma: ", self.gamma)
        print("Number of Episodes: ", self.n_episodes)
        print("Batch size: ", self.batch_size)
        print("Learning Rate: ", self.learning_rate)
        print("Inputs: ", self.n_inputs)
        print("hidden Layers: ", self.n_hidden)
        print("Number of Actions: ", self.n_actions)
        print("Model: ", self.q_network.summary())
        print("Input Shape: ", self.input_shape)
        print("Output Shape: ", self.output_shape)
        print("Trainable Parameters: ", self.trainable_params)
        print("lr_schedule: ", {"initial_learning_rate": self.lr_schedule.initial_learning_rate,
                                "decay_steps": self.lr_schedule.decay_steps,
                                "decay_rate": self.lr_schedule.decay_rate})
        print("----------------------")

        # Metrics ---------------------------
        self.episode_reward_history = []
        self.episode_length_history = []

    def build_network(self):
        """Generates a Keras model for the Q-function approximator."""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.n_inputs,)))
        for nodes in self.n_hidden:
            model.add(tf.keras.layers.Dense(nodes, activation='relu'))
        model.add(tf.keras.layers.Dense(self.n_actions, activation='linear'))
        return model

    def hard_update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def take_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_actions)
        q_values = self.q_network.predict(np.expand_dims(state, axis=0))
        return np.argmax(q_values[0])

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def sample_experiences(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def calculate_target(self, rewards, next_states, dones):
        target_q = self.target_network.predict(next_states)
        return rewards + (1 - dones) * self.gamma * np.max(target_q, axis=1)

    def train_q_network(self, states, actions, target_vals):
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions)
            q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            loss = tf.reduce_mean(tf.square(target_vals - q_values))
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def train(self):
        """Train the agent."""
        for episode in range(self.n_episodes):
            state = self.env.reset()
            state = np.reshape(state, [self.n_inputs])
            total_reward = 0
            total_steps = 0

            while True:
                action = self.take_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [self.n_inputs])

                self.store_experience(state, action, reward, next_state, done)

                total_reward += reward
                total_steps += 1

                if done:
                    break

                state = next_state

                if len(self.replay_memory) >= self.batch_size:
                    states, actions, rewards, next_states, dones = self.sample_experiences()
                    target_vals = self.calculate_target(rewards, next_states, dones)
                    self.train_q_network(states, actions, target_vals)

                    if total_steps % self.steps_per_target_update == 0:
                        self.hard_update_target_network()

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.decay_epsilon

            self.episode_reward_history.append(total_reward)
            self.episode_length_history.append(total_steps)

            if (episode + 1) % 10 == 0:
                avg_rewards = np.mean(self.episode_reward_history[-10:])
                print(f"Episode {episode + 1}/{self.n_episodes}, average last 10 episode rewards: {avg_rewards}")

                if avg_rewards >= 500.0:
                    break

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



    def plot_rewards(self):
        """Plot the episode reward history."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_reward_history)
        plt.xlabel('Episode')
        plt.ylabel('Collected rewards')
        plt.show()


if __name__ == '__main__':
    agent = DeepQLearningClassical()
    agent.train()
    agent.plot_rewards()


