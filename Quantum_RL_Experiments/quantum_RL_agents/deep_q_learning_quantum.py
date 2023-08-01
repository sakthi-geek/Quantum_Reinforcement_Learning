
# Update package resources to account for version changes.
import importlib, pkg_resources
importlib.reload(pkg_resources)

import tensorflow as tf
import tensorflow_quantum as tfq

import gym, cirq, sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
tf.get_logger().setLevel('ERROR')

import random
from Quantum_RL_Experiments.helper import Rescaling, ReUploadingPQC

class DeepQLearningQuantum:
    def __init__(self, n_qubits=4, n_layers=5, n_actions=2, gamma=0.99, n_episodes=2000, batch_size=16):
        # Hyperparameters
        self.batch_size = batch_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.ops = [cirq.Z(q) for q in self.qubits]
        self.observables = [self.ops[0] * self.ops[1],
                            self.ops[2] * self.ops[3]]  # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1

        # Initialize models
        self.model = self.generate_model_Qlearning(False)
        self.model_target = self.generate_model_Qlearning(True)
        self.model_target.set_weights(self.model.get_weights())

        tf.keras.utils.plot_model(self.model, show_shapes=True, dpi=70)
        tf.keras.utils.plot_model(self.model_target, show_shapes=True, dpi=70)

        # Initialize replay memory and other training-related variables
        # Define replay memory
        self.max_memory_length = 10000  # Maximum replay length
        self.replay_memory = deque(maxlen=self.max_memory_length)
        self.epsilon = 1.0 # Epsilon greedy parameter
        self.epsilon_min = 0.01 # Minimum epsilon greedy parameter
        self.decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter
        self.episode_reward_history = []
        self.steps_per_update = 10  # Train the model every x steps
        self.steps_per_target_update = 30  # Update the target model every x steps

        # Prepare the optimizers
        self.optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
        self.optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
        self.optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

        # Assign the model parameters to each optimizer
        self.w_in, self.w_var, self.w_out = 1, 0, 2

        # Set up the environment
        self.env = gym.make("CartPole-v1")

        self.episode_reward_history = []

        # ... further initialization as needed ...

    def generate_model_Qlearning(self, target):  ##(qubits, n_layers, n_actions, observables, target)
        """Generates a Keras model for a data re-uploading PQC Q-function approximator."""

        input_tensor = tf.keras.Input(shape=(len(self.qubits),), dtype=tf.dtypes.float32, name='input')
        re_uploading_pqc = ReUploadingPQC(self.qubits, self.n_layers, self.observables, activation='tanh')([input_tensor])
        process = tf.keras.Sequential([Rescaling(len(self.observables))], name=target * "Target" + "Q-values")
        Q_values = process(re_uploading_pqc)
        model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)

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
            loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        for optimizer, w in zip([self.optimizer_in, self.optimizer_var, self.optimizer_out], [self.w_in, self.w_var, self.w_out]):
            optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])

    def sample_interaction(self, state):
        """Sample an interaction with the environment."""
        interaction = self.interact_env(state)
        self.replay_memory.append(interaction)
        return interaction['next_state'], interaction['reward'], interaction['done']

    def update_models(self, step_count):
        """Update the Q-learning and target models at appropriate intervals."""
        if step_count % self.steps_per_update == 0:
            # Create training batch and update Q model
            training_batch = random.sample(self.replay_memory, self.batch_size)
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
        step_count = 0
        for episode in range(self.n_episodes):
            episode_reward = 0
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


