import tensorflow as tf
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from functools import reduce
import random


# You may add import statements for your helper functions here if needed

class DeepQLearningClassical:
    def __init__(self, n_actions=2, gamma=0.99, n_episodes=2000, batch_size=16, learning_rate=0.01):
        # Hyperparameters
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.input_shape = 4  # Set to the number of environment states. For CartPole, it's 4.

        # Initialize models
        self.model = self.generate_model()
        self.model_target = self.generate_model()
        self.model_target.set_weights(self.model.get_weights())

        tf.keras.utils.plot_model(self.model, show_shapes=True, dpi=70)
        tf.keras.utils.plot_model(self.model_target, show_shapes=True, dpi=70)

        # Initialize replay memory and other variables
        self.max_memory_length = 10000
        self.replay_memory = deque(maxlen=self.max_memory_length)
        self.epsilon = 1.0 # Epsilon greedy parameter
        self.epsilon_min = 0.01 # Minimum epsilon greedy parameter
        self.decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter
        self.steps_per_update = 10
        self.steps_per_target_update = 30
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Environment
        self.env = gym.make("CartPole-v1")
        self.episode_reward_history = []


    def generate_model(self):
        """Generates a Keras model for classical Q-learning."""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_shape,)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(self.n_actions, activation='linear'))
        model.compile(optimizer=self.optimizer, loss='mse')
        return model

    def interact_env(self, state):
        """Interaction with the environment."""
        state_array = np.array(state)
        state = tf.convert_to_tensor([state_array])

        coin = np.random.random()
        if coin > self.epsilon:
            q_vals = self.model(state)
            action = int(tf.argmax(q_vals[0]).numpy())
        else:
            action = np.random.choice(self.n_actions)

        next_state, reward, done, _ = self.env.step(action)
        interaction = {'state': state_array, 'action': action, 'next_state': next_state.copy(),
                       'reward': reward, 'done': np.float32(done)}

        return interaction

    @tf.function
    def Q_learning_update(self, states, actions, rewards, next_states, done):
        """Q-learning model update."""
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        done = tf.convert_to_tensor(done)

        future_rewards = self.model_target(next_states)
        target_q_values = rewards + (self.gamma * tf.reduce_max(future_rewards, axis=1)
                                     * (1.0 - done))
        masks = tf.one_hot(actions, self.n_actions)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def sample_interaction(self, state):
        interaction = self.interact_env(state)
        self.replay_memory.append(interaction)
        return interaction['next_state'], interaction['reward'], interaction['done']

    def update_models(self, step_count):
        if step_count % self.steps_per_update == 0:
            training_batch = np.random.choice(self.replay_memory, size=self.batch_size)
            self.update_Q_model(training_batch)
        if step_count % self.steps_per_target_update == 0:
            self.model_target.set_weights(self.model.get_weights())

    def update_Q_model(self, training_batch):
        self.Q_learning_update(
            np.asarray([x['state'] for x in training_batch]),
            np.asarray([x['action'] for x in training_batch]),
            np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
            np.asarray([x['next_state'] for x in training_batch]),
            np.asarray([x['done'] for x in training_batch], dtype=np.float32))

    def train(self):
        step_count = 0
        for episode in range(self.n_episodes):
            episode_reward = 0
            state = self.env.reset()

            while True:
                state, reward, done = self.sample_interaction(state)
                episode_reward += reward
                step_count += 1

                self.update_models(step_count)

                if done:
                    break

            self.epsilon = max(self.epsilon * self.decay_epsilon, self.epsilon_min)
            self.episode_reward_history.append(episode_reward)
            if (episode + 1) % 10 == 0:
                avg_rewards = np.mean(self.episode_reward_history[-10:])
                print("Episode {}/{}, average last 10 rewards {}".format(
                    episode + 1, self.n_episodes, avg_rewards))

                if avg_rewards >= 500.0:
                    break

        self.plot_rewards()

    def save_model(self, path):
        """Save the model to the specified path."""
        self.model.save(path)

    def load_model(self, path):
        """Load the model from the specified path."""
        self.model = tf.keras.models.load_model(path)
        # Update the target model as well
        self.model_target.set_weights(self.model.get_weights())

    def plot_rewards(self):
        """Plot the episode reward history of the agent."""
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
                action = np.argmax(self.model.predict(state[None, :]))
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        if render:
            self.env.close()



###------------------------------------------------------------------------------
    # import tensorflow as tf
    # import numpy as np
    # import gym
    # from collections import deque
    # import random
    # import matplotlib.pyplot as plt
    #
    # class DeepQLearningClassical:
    #     def __init__(self, n_actions=2, gamma=0.99, n_episodes=2000, batch_size=16, learning_rate=0.01):
    #         # Hyperparameters
    #         self.n_actions = n_actions
    #         self.gamma = gamma
    #         self.n_episodes = n_episodes
    #         self.batch_size = batch_size
    #         self.learning_rate = learning_rate
    #
    #         # Initialize models
    #         self.model = self.generate_model()
    #         self.model_target = self.generate_model()
    #         self.model_target.set_weights(self.model.get_weights())
    #
    #         tf.keras.utils.plot_model(self.model, show_shapes=True, dpi=70)
    #         tf.keras.utils.plot_model(self.model_target, show_shapes=True, dpi=70)
    #
    #         # Initialize replay memory and other training-related variables
    #         self.max_memory_length = 10000  # Maximum replay length
    #         self.replay_memory = deque(maxlen=self.max_memory_length)
    #         self.epsilon = 1  # Epsilon greedy parameter - start with a high value to encourage exploration
    #         self.epsilon_min = 0.01  # Minimum epsilon greedy parameter
    #         self.decay_epsilon = 0.99  # Decay rate of epsilon greedy parameter - decay it to encourage exploitation
    #         self.episode_reward_history = []
    #         self.steps_per_update = 10  # Train the model every x steps
    #         self.steps_per_target_update = 30  # Update the target model every x steps
    #
    #         self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=True)
    #
    #         # Set up the environment
    #         self.env = gym.make("CartPole-v1")
    #
    #         self.episode_reward_history = []
    #
    #     def generate_model(self):
    #         """Generates a Keras model for a classical Q-function approximator."""
    #         model = tf.keras.Sequential([
    #             tf.keras.layers.Input(shape=(4,)),  # Input for CartPole's state space shape
    #             tf.keras.layers.Dense(24, activation='relu'),
    #             tf.keras.layers.Dense(24, activation='relu'),
    #             tf.keras.layers.Dense(self.n_actions)  # Output the Q-values for each action
    #         ])
    #         return model
    #
    #     def interact_env(self, state):
    #         """Performs an interaction step in the environment."""
    #         # Sample action
    #         coin = np.random.random()
    #         if coin > self.epsilon:
    #             q_vals = self.model(state[None, :])
    #             action = int(tf.argmax(q_vals[0]).numpy())
    #         else:
    #             action = np.random.choice(self.n_actions)
    #
    #         # Apply sampled action in the environment, receive reward and next state
    #         next_state, reward, done, _ = self.env.step(action)
    #
    #         interaction = {'state': state, 'action': action, 'next_state': next_state.copy(),
    #                        'reward': reward, 'done': np.float32(done)}
    #
    #         return interaction
    #
    #     @tf.function
    #     def Q_learning_update(self, states, actions, rewards, next_states, done):
    #         """Update the Q-learning model based on a batch of interactions."""
    #         # Compute their target q_values and the masks on sampled actions
    #         future_rewards = self.model_target(next_states)
    #         target_q_values = rewards + (self.gamma * tf.reduce_max(future_rewards, axis=1)
    #                                      * (1.0 - done))
    #         masks = tf.one_hot(actions, self.n_actions)
    #
    #         # Train the model on the states and target Q-values
    #         with tf.GradientTape() as tape:
    #             q_values = self.model(states)
    #             q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
    #             loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)
    #
    #         # Backpropagation
    #         grads = tape.gradient(loss, self.model.trainable_variables)
    #         self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    #
    # def sample_interaction(self, state):
    #     """Sample an interaction with the environment."""
    #     interaction = self.interact_env(state)
    #     self.replay_memory.append(interaction)
    #     return interaction['next_state'], interaction['reward'], interaction['done']
    #
    # def update_models(self, step_count):
    #     """Update the Q-learning and target models at appropriate intervals."""
    #     if step_count % self.steps_per_update == 0:
    #         # Create training batch and update Q model
    #         training_batch = np.random.choice(self.replay_memory, self.batch_size)
    #         self.update_Q_model(training_batch)
    #     if step_count % self.steps_per_target_update == 0:
    #         self.model_target.set_weights(self.model.get_weights())
    #
    # def update_Q_model(self, training_batch):
    #     """Perform a Q-learning update using a batch of training data."""
    #     self.Q_learning_update(
    #         np.asarray([x['state'] for x in training_batch]),
    #         np.asarray([x['action'] for x in training_batch]),
    #         np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
    #         np.asarray([x['next_state'] for x in training_batch]),
    #         np.asarray([x['done'] for x in training_batch], dtype=np.float32))

