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


#-----------------------------------------------------------------------------------
#
# import tensorflow as tf
# import tensorflow_quantum as tfq
# import gym
# import cirq
# import sympy
# import numpy as np
# from collections import defaultdict
# from functools import reduce
# import matplotlib.pyplot as plt
# from cirq.contrib.svg import SVGCircuit
# from Quantum_RL_Experiments.helper import ReUploadingPQC, Alternating, Rescaling
#
#
# class ActorCriticQuantum:
#     def __init__(self, env_name="CartPole-v1", n_qubits=4, n_layers=5, n_actions=2, gamma=0.99, n_episodes=2000,
#                  max_steps_per_episode=10000, learning_rate=0.01, state_bounds=np.array([2.4, 2.5, 0.21, 2.5])):
#
#         self.env_name = env_name
#         self.env = gym.make(env_name)
#         self.gamma = gamma
#         self.n_episodes = n_episodes
#         self.max_steps_per_episode = max_steps_per_episode
#         self.learning_rate = learning_rate
#         self.state_bounds = state_bounds
#
#         self.n_qubits = n_qubits
#         self.n_layers = n_layers
#         self.n_actions = n_actions
#
#         self.actor_qubits = cirq.GridQubit.rect(1, self.n_qubits)
#         self.actor_ops = [cirq.Z(q) for q in self.actor_qubits]
#         self.actor_observables = [reduce((lambda x, y: x * y), self.actor_ops)]  # Z_0*Z_1*Z_2*Z_3
#         # self.actor_observables = [cirq.Z(self.qubits[i]) for i in range(self.n_qubits)]
#
#         self.critic_qubits = cirq.GridQubit.rect(1, self.n_qubits)
#         self.critic_ops = [cirq.Z(q) for q in self.critic_qubits]
#         self.critic_observables = [self.critic_ops[0] * self.critic_ops[1],
#                             self.critic_ops[2] * self.critic_ops[3]]  # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1
#         # self.critic_observables = [cirq.Z(self.qubits[i]) for i in range(self.n_qubits)]
#
#         self.beta = 1.0
#
#         # Actor Model
#         self.actor_model = self.generate_actor_model()
#         # # Critic Model
#         self.critic_model = self.generate_critic_model()
#
#         # self.actor_model = self.generate_model_value(self.qubits, n_layers, n_actions, self.observables)
#         # self.critic_model = self.generate_model_value(self.qubits, n_layers, 1, self.observables)
#
#         # Prepare the actor optimizers
#         self.actor_optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
#         self.actor_optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
#         self.actor_optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
#
#         # Prepare the critic optimizers
#         self.critic_optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
#         self.critic_optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
#         self.critic_optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
#
#         # Assign the model parameters to each optimizer
#         self.w_in, self.w_var, self.w_out = 1, 0, 2
#
#         self.huber_loss = tf.keras.losses.Huber()
#         self.episode_reward_history = []
#         print("----------------------")
#
#     def generate_actor_model(self):
#         input_tensor = tf.keras.Input(shape=(len(self.actor_qubits),), dtype=tf.float32, name='input')
#         re_uploading_pqc = ReUploadingPQC(self.actor_qubits, self.n_layers, self.actor_observables)([input_tensor])
#         action = tf.keras.Sequential([
#             Alternating(self.n_actions),
#             tf.keras.layers.Lambda(lambda x: x * self.beta),
#             tf.keras.layers.Softmax()
#         ], name="observables-policy")(re_uploading_pqc)
#         return tf.keras.Model(inputs=[input_tensor], outputs=action)
#
#     def generate_critic_model(self):
#         input_tensor = tf.keras.Input(shape=(len(self.critic_qubits),), dtype=tf.float32, name='input')
#         re_uploading_pqc = ReUploadingPQC(self.critic_qubits, self.n_layers, self.critic_observables)([input_tensor])
#         critic = tf.keras.Sequential([
#             Rescaling(len(self.critic_observables))
#         ], name="Q-values")(re_uploading_pqc)
#         return tf.keras.Model(inputs=[input_tensor], outputs=critic)
#
#     # def generate_pqc_model(self):
#     #     input_tensor = tf.keras.Input(shape=(self.n_qubits,), dtype=tf.float32, name='input')
#     #     re_uploading_pqc = ReUploadingPQC(self.qubits, self.n_layers, self.observables, activation='tanh')([input_tensor])
#     #     process = tf.keras.Sequential([Rescaling(len(self.observables))])
#     #     output = process(re_uploading_pqc)
#     #     model = tf.keras.Model(inputs=[input_tensor], outputs=[output])
#     #     return model
#
#     def train(self):
#
#         episode_count = 0
#         running_reward = 0
#         N = 10  # You can adjust this to capture metrics every N episodes
#         best_weights = None  # Store the best model weights
#
#         for _ in range(self.n_episodes):
#             state = self.env.reset()
#             action_probs_history = []
#             critic_value_history = []
#             rewards_history = []
#             episode_reward = 0
#             # episode_entropy = []
#             # episode_advantages = []
#
#             with tf.GradientTape() as tape:
#                 for timestep in range(1, self.max_steps_per_episode):
#                     state = tf.convert_to_tensor(state)
#                     state = tf.expand_dims(state, 0)
#                     action_probs = self.actor_model(state)
#                     critic_value = self.critic_model(state)
#                     critic_value_history.append(critic_value[0, 0])
#                     action = np.random.choice(self.n_actions, p=np.squeeze(action_probs))
#                     action_probs_history.append(tf.math.log(action_probs[0, action]))
#
#                     state, reward, done, _ = self.env.step(action)
#                     rewards_history.append(reward)
#                     episode_reward += reward
#
#                     if done:
#                         break
#
#                     # # Update environment-specific metrics - GRANULAR
#                     # if episode_count % N == 0:
#                     #     pole_angle, pole_velocity, cart_position, cart_velocity = state
#                     #     self.pole_angle_history.append(pole_angle)
#                     #     self.pole_angular_velocity_history.append(pole_velocity)
#                     #     self.cart_position_history.append(cart_position)
#                     #     self.cart_velocity_history.append(cart_velocity)
#                     #     self.action_distribution[action] += 1
#                     #
#                     #     # Entropy of the policy
#                     #     entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs))
#                     #     episode_entropy.append(entropy.numpy())
#                     #
#                     # # Update termination reasons (only if done)
#                     # if done:
#                     #     # Store the episode length
#                     #     self.episode_length_history.append(timestep)
#                     #     if abs(pole_angle) > 0.2094:  # ~12 degrees
#                     #         self.termination_reasons["pole_angle"] += 1
#                     #     elif abs(cart_position) > 2.4:
#                     #         self.termination_reasons["cart_position"] += 1
#                     #     break
#
#                 running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
#                 self.episode_reward_history.append(episode_reward)
#                 returns = []
#                 discounted_sum = 0
#                 for r in rewards_history[::-1]:
#                     discounted_sum = r + self.gamma * discounted_sum
#                     returns.insert(0, discounted_sum)
#
#                 returns = np.array(returns)
#                 returns = (returns - np.mean(returns)) / (np.std(returns) + np.finfo(np.float32).eps.item())
#                 returns = returns.tolist()
#
#                 # Calculating loss values to update our network
#                 history = zip(action_probs_history, critic_value_history, returns)
#                 actor_losses = []
#                 critic_losses = []
#                 for log_prob, value, ret in history:
#                     diff = ret - value
#                     actor_losses.append(-log_prob * diff)
#                     critic_losses.append(self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))
#
#                 loss_value = sum(actor_losses) + sum(critic_losses)
#
#                 actor_grads = tape.gradient(loss_value, self.actor_model.trainable_variables)
#                 for optimizer, w in zip([self.actor_optimizer_in, self.actor_optimizer_var, self.actor_optimizer_out],
#                                             [self.w_in, self.w_var, self.w_out]):
#                         optimizer.apply_gradients([(actor_grads[w], self.actor_model.trainable_variables[w])])
#
#                 critic_grads = tape.gradient(loss_value, self.critic_model.trainable_variables)
#                 for optimizer, w in zip([self.critic_optimizer_in, self.critic_optimizer_var, self.critic_optimizer_out],
#                                             [self.w_in, self.w_var, self.w_out]):
#                         optimizer.apply_gradients([(critic_grads[w], self.critic_model.trainable_variables[w])])
#
#
#                 # Clear the loss and reward history
#                 action_probs_history.clear()
#                 critic_value_history.clear()
#                 rewards_history.clear()
#
#                 # # Storing actor and critic losses separately
#                 # self.actor_loss_history.append(np.mean(actor_losses))
#                 # self.critic_loss_history.append(np.mean(critic_losses))
#
#                 # Calculate average entropy and advantage values and store them
#                 # self.entropy_history.append(np.mean(episode_entropy))
#
#                 # Storing advantage values
#                 # advantages = [ret - value for ret, value in zip(returns, critic_value_history)]
#                 # self.advantage_values_history.append(np.mean(advantages))
#
#                 # Gradient magnitudes
#                 # gradient_magnitudes = [tf.norm(grad).numpy() for grad in grads if grad is not None]
#                 # self.gradient_magnitudes_history.append(np.mean(gradient_magnitudes))
#
#             # # Storing rewards stats - GRANULAR
#             # if episode_count % N == 0:
#             #     self.episode_rewards_min.append(np.min(rewards_history))
#             #     self.episode_rewards_max.append(np.max(rewards_history))
#             #     self.episode_rewards_mean.append(np.mean(rewards_history))
#             #     self.episode_rewards_std.append(np.std(rewards_history))
#
#             # # Early stopping check
#             # if len(self.episode_reward_history) >= self.patience:
#             #     avg_last_rewards = np.mean(self.episode_reward_history[-self.patience:])
#             #     if avg_last_rewards - self.best_avg_reward >= self.min_improvement:
#             #         self.best_avg_reward = avg_last_rewards
#             #         self.waited_episodes = 0
#             #         best_weights = self.model.get_weights()  # Save the best weights
#             #     else:
#             #         self.waited_episodes += 1
#             #         if self.waited_episodes >= self.patience:
#             #             print(f"Early stopping after {self.waited_episodes} episodes without improvement.")
#             #             self.model.set_weights(best_weights)  # Restore the best weights
#             #             break
#
#             episode_count += 1
#             if episode_count % 10 == 0:
#                 avg_rewards = np.mean(self.episode_reward_history[-10:])
#                 print(
#                     "Episode {}/{}, average last 10 rewards {}".format(episode_count, self.n_episodes, avg_rewards))
#
#                 if avg_rewards >= 500.0:
#                     break
#
#         self.plot_rewards()
#
#
#     def save_model(self, path):
#         """Save the model to the specified path."""
#         self.model.save(path)
#
#     def load_model(self, path):
#         """Load the model from the specified path."""
#         self.model = tf.keras.models.load_model(path)
#
#     def plot_rewards(self):
#         plt.figure(figsize=(10, 5))
#         plt.plot(self.episode_reward_history)
#         plt.xlabel('Epsiode')
#         plt.ylabel('Collected rewards')
#         plt.show()
#
#     def play(self, episodes=5, render=False):
#         for episode in range(episodes):
#             state = self.env.reset()
#             done = False
#             total_reward = 0
#             while not done:
#                 if render:
#                     self.env.render()
#                 action_probs, _ = self.model(state[None, :])
#                 action = np.argmax(action_probs[0])
#                 state, reward, done, _ = self.env.step(action)
#                 total_reward += reward
#             print(f"Episode {episode + 1}: Total Reward = {total_reward}")
#
#         if render:
#             self.env.close()
#
#
#
# if __name__ == "__main__":
#     actor_critic_agent = ActorCriticQuantum()
#     actor_critic_agent.train()
#     actor_critic_agent.play(render=True)
