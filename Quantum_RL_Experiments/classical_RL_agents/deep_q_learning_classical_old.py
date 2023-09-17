
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

        # Initialize models
        self.model = self.generate_model_classical()
        self.model_target = self.generate_model_classical()
        self.model_target.set_weights(self.model.get_weights())

        tf.keras.utils.plot_model(self.model, show_shapes=True, dpi=70)
        tf.keras.utils.plot_model(self.model_target, show_shapes=True, dpi=70)

        self.input_shape = self.model.input_shape
        self.output_shape = self.model.output_shape
        self.trainable_params = self.model.count_params()

        # Initialize replay memory and other variables
        self.max_memory_length = 10000
        self.replay_memory = deque(maxlen=self.max_memory_length)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter - range - 0 to 1 - 1 means decay is not happening - 0 means decay is happening very fast
        self.steps_per_update = 10
        self.steps_per_target_update = 30


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
        print("Model: ", self.model.summary())
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

    def generate_model_classical(self):
        """Generates a Keras model for a classical Q-function approximator."""
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.n_inputs,)))  # Replace 4 with the dimension of your state space
        for nodes in self.n_hidden:
            model.add(tf.keras.layers.Dense(nodes, activation='relu'))
        model.add(tf.keras.layers.Dense(self.n_actions, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model


    # Defining a function that performs an interaction step in the environment
    def interact_env(self, state):  ##(state, model, epsilon, n_actions, env)
        """Performs an interaction step in the environment."""
        # Preprocess state
        state_array = np.array(state)
        state = tf.convert_to_tensor([state_array])

        # Sample action
        coin = np.random.random()
        if coin > self.epsilon:
            q_vals = self.model([state])
            action = int(tf.argmax(q_vals[0]).numpy())
        else:
            action = np.random.choice(self.n_actions)

        # Apply sampled action in the environment, receive reward and next state
        next_state, reward, done, _ = self.env.step(action)

        interaction = {'state': state_array, 'action': action, 'next_state': next_state.copy(),
                       'reward': reward, 'done': np.float32(done)}

        return interaction

    # Function that updates the Q-function using a batch of interactions
    @tf.function
    def Q_learning_update(self, states, actions, rewards, next_states, done):
        """Update the Q-learning model based on a batch of interactions."""
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        done = tf.convert_to_tensor(done)

        # Compute their target q_values and the masks on sampled actions
        future_rewards = self.model_target([next_states])
        target_q_values = rewards + (self.gamma * tf.reduce_max(future_rewards, axis=1)
                                     * (1.0 - done))
        masks = tf.one_hot(actions, self.n_actions)

        # Train the model on the states and target Q-values
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            q_values = self.model([states])
            q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values_masked))

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def sample_interaction(self, state):
        """Sample an interaction with the environment."""
        interaction = self.interact_env(state)
        self.replay_memory.append(interaction)
        return interaction['next_state'], interaction['reward'], interaction['done']

    def update_models(self, step_count):
        """Update the Q-learning and target models at appropriate intervals."""
        if step_count % self.steps_per_update == 0:
            # Create training batch and update Q model
            training_batch = np.random.choice(self.replay_memory, size=self.batch_size)
            self.update_Q_model(training_batch)
        if step_count % self.steps_per_target_update == 0:
            self.model_target.set_weights(self.model.get_weights())

    def update_Q_model(self, training_batch):
        """Perform a Q-learning update using a batch of training data."""
        self.Q_learning_update(
            np.asarray([x['state'] for x in training_batch]),
            np.asarray([x['action'] for x in training_batch]),
            np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
            np.asarray([x['next_state'] for x in training_batch]),
            np.asarray([x['done'] for x in training_batch], dtype=np.float32))

    def train(self):
        """Train the agent."""
        for episode in range(self.n_episodes):
            episode_reward = 0
            step_count = 0
            state = self.env.reset()

            while True:
                # Sample interaction with the environment
                state, reward, done = self.sample_interaction(state)
                episode_reward += reward
                step_count += 1

                # Update Q-learning model and target model
                self.update_models(step_count)

                if done:
                    break

            # Exponentially decay epsilon
            self.epsilon = max(self.epsilon * self.decay_epsilon, self.epsilon_min)

            # # Linear epsilon decay
            # self.epsilon -= (1 - self.epsilon_min) / self.n_episodes
            # self.epsilon = max(self.epsilon, self.epsilon_min)

            self.episode_reward_history.append(episode_reward)
            self.episode_length_history.append(step_count)
            if (episode + 1) % 10 == 0:
                avg_rewards = np.mean(self.episode_reward_history[-10:])
                print("Episode {}/{}, average last 10 rewards {}".format(
                    episode + 1, self.n_episodes, avg_rewards))
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


