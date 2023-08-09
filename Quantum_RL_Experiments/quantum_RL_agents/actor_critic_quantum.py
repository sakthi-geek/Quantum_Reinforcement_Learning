import tensorflow as tf
import tensorflow_quantum as tfq
import gym
import cirq
import sympy
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
from Quantum_RL_Experiments.helper import ReUploadingPQC, Alternating


class ActorCriticQuantum:
    def __init__(self, env_name="CartPole-v1", n_qubits=4, n_layers=5, n_actions=2, gamma=0.99, n_episodes=2000,
                 max_steps_per_episode=10000, learning_rate=0.01, state_bounds=np.array([2.4, 2.5, 0.21, 2.5])):

        self.env_name = env_name
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = learning_rate
        self.state_bounds = state_bounds

        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.observables = [cirq.Z(q) for q in self.qubits]

        self.actor_model = self.generate_model(self.qubits, n_layers, n_actions, self.observables)
        self.critic_model = self.generate_model(self.qubits, n_layers, 1, self.observables)

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.huber_loss = keras.losses.Huber()
        self.episode_reward_history = []

    # Reusing code for generating quantum circuit, same as PolicyGradientQuantum
    def one_qubit_rotation(self, qubit, symbols):
        """
        Returns Cirq gates that apply a rotation of the bloch sphere about the X,
        Y and Z axis, specified by the values in `symbols`.
        """
        return [cirq.rx(symbols[0])(qubit),
                cirq.ry(symbols[1])(qubit),
                cirq.rz(symbols[2])(qubit)]

    def entangling_layer(self, qubits):
        """
        Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
        """
        cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
        cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
        return cz_ops

    def generate_circuit(self, qubits, n_layers):
        """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
        # Number of qubits
        n_qubits = len(qubits)

        # Sympy symbols for variational angles
        params = sympy.symbols(f'theta(0:{3 * (n_layers + 1) * n_qubits})')
        params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

        # Sympy symbols for encoding angles
        inputs = sympy.symbols(f'x(0:{n_layers})' + f'_(0:{n_qubits})')
        inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

        # Define circuit
        circuit = cirq.Circuit()
        for l in range(n_layers):
            # Variational layer
            circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
            circuit += entangling_layer(qubits)
            # Encoding layer
            circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

        # Last varitional layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i, q in enumerate(qubits))

        return circuit, list(params.flat), list(inputs.flat)

    def generate_model_policy(self, qubits, n_layers, n_actions, beta, observables):
        """Generates a Keras model for a data re-uploading PQC policy."""

        input_tensor = tf.keras.Input(shape=(len(qubits),), dtype=tf.dtypes.float32, name='input')
        re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables)([input_tensor])
        process = tf.keras.Sequential([
            Alternating(n_actions),
            tf.keras.layers.Lambda(lambda x: x * beta),
            tf.keras.layers.Softmax()
        ], name="observables-policy")
        policy = process(re_uploading_pqc)
        model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

        return model

    def generate_model_value(self, qubits, n_layers, observables):
        """Generates a Keras model for a data re-uploading PQC value function."""

        input_tensor = tf.keras.Input(shape=(len(qubits),), dtype=tf.dtypes.float32, name='input')
        re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables)([input_tensor])
        value = tf.keras.layers.Dense(1)(re_uploading_pqc)
        model = tf.keras.Model(inputs=[input_tensor], outputs=value)

        return model

    def train(self):
        episode_count = 0
        running_reward = 0

        for _ in range(self.n_episodes):
            state = self.env.reset()
            action_probs_history = []
            critic_value_history = []
            rewards_history = []
            episode_reward = 0

            with tf.GradientTape() as tape:
                for timestep in range(1, self.max_steps_per_episode):
                    state_normalized = state / self.state_bounds
                    state_tensor = tf.convert_to_tensor(state_normalized)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs, critic_value = self.actor_model(state_tensor)
                    critic_value_history.append(critic_value[0, 0])
                    action = np.random.choice(self.env.action_space.n, p=np.squeeze(action_probs))
                    action_probs_history.append(tf.math.log(action_probs[0, action]))
                    state, reward, done, _ = self.env.step(action)
                    rewards_history.append(reward)
                    episode_reward += reward
                    if done:
                        break

                running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
                self.episode_reward_history.append(episode_reward)
                returns = []
                discounted_sum = 0
                for r in rewards_history[::-1]:
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + np.finfo(np.float32).eps.item())
                returns = returns.tolist()

                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in zip(action_probs_history, critic_value_history, returns):
                    diff = ret - value
                    actor_losses.append(-log_prob * diff)
                    critic_losses.append(self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value,
                                      self.actor_model.trainable_variables + self.critic_model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.actor_model.trainable_variables + self.critic_model.trainable_variables))

            episode_count += 1
            if episode_count % 10 == 0:
                avg_rewards = np.mean(self.episode_reward_history[-10:])
                print(
                    "Episode {}/{}, average last 10 rewards {}".format(episode_count, self.n_episodes, avg_rewards))
                if avg_rewards >= 195:
                    break

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
                action_probs, _ = self.model(state[None, :])
                action = np.argmax(action_probs[0])
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        if render:
            self.env.close()
