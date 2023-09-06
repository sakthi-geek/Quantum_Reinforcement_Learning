import tensorflow as tf
import tensorflow_quantum as tfq
import gym
import cirq
import sympy
import numpy as np
from collections import defaultdict
from functools import reduce
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
from Quantum_RL_Experiments.helper import ReUploadingPQC, Alternating, Rescaling


eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

class ActorCriticQuantum:
    def __init__(self, env_name="CartPole-v1", n_qubits=4, n_layers=5, n_actions=2, gamma=0.99, n_episodes=3000,
                 max_steps_per_episode=10000, learning_rate=0.01, state_bounds=np.array([2.4, 2.5, 0.21, 2.5])):

        self.env_name = env_name
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = learning_rate
        self.state_bounds = state_bounds

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions

        self.actor_qubits = cirq.GridQubit.rect(1, self.n_qubits)
        self.actor_ops = [cirq.Z(q) for q in self.actor_qubits]
        self.actor_observables = [reduce((lambda x, y: x * y), self.actor_ops)]  # Z_0*Z_1*Z_2*Z_3
        # self.actor_observables = [cirq.Z(self.qubits[i]) for i in range(self.n_qubits)]

        self.critic_qubits = cirq.GridQubit.rect(1, self.n_qubits)
        self.critic_ops = [cirq.Z(q) for q in self.critic_qubits]
        self.critic_observables = [self.critic_ops[0] * self.critic_ops[1],
                            self.critic_ops[2] * self.critic_ops[3]]  # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1
        # self.critic_observables = [cirq.Z(self.qubits[i]) for i in range(self.n_qubits)]

        self.beta = 1.0

        # Shared Model---------------------------------------------------
        # self.shared_actor_critic_model = self.generate_shared_actor_critic_model()
        # # Prepare the shared optimizers
        # self.shared_optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
        # self.shared_optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
        # self.shared_optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
        #----------------------------------------------------------------

        # Actor Model
        self.actor_model = self.generate_actor_model()
        # # Critic Model
        self.critic_model = self.generate_critic_model()

        self.input_shape = [self.actor_model.input_shape, self.critic_model.input_shape]
        self.output_shape = [self.actor_model.output_shape, self.critic_model.output_shape]
        self.trainable_params = [self.actor_model.count_params(), self.critic_model.count_params()]
        print(self.input_shape, self.output_shape, self.trainable_params)

        # Prepare the actor optimizers
        self.actor_optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
        self.actor_optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
        self.actor_optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

        # Prepare the critic optimizers
        self.critic_optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
        self.critic_optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
        self.critic_optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

        # Assign the model parameters to each optimizer
        self.w_in, self.w_var, self.w_out = 1, 0, 2

        self.huber_loss = tf.keras.losses.Huber()

        # Metrics ---------------------------
        self.episode_reward_history = []
        self.episode_length_history = []

        print("----------------------")

    def generate_actor_model(self):
        input_tensor = tf.keras.Input(shape=(len(self.actor_qubits),), dtype=tf.float32, name='input')
        re_uploading_pqc = ReUploadingPQC(self.actor_qubits, self.n_layers, self.actor_observables)([input_tensor])
        action = tf.keras.Sequential([
            Alternating(self.n_actions),
            tf.keras.layers.Lambda(lambda x: x * self.beta),
            tf.keras.layers.Softmax()
        ], name="observables-policy")(re_uploading_pqc)
        return tf.keras.Model(inputs=[input_tensor], outputs=action)

    def generate_critic_model(self):
        input_tensor = tf.keras.Input(shape=(len(self.critic_qubits),), dtype=tf.float32, name='input')
        re_uploading_pqc = ReUploadingPQC(self.critic_qubits, self.n_layers, self.critic_observables)([input_tensor])
        critic = tf.keras.Sequential([
            Rescaling(len(self.critic_observables))
        ], name="Q-values")(re_uploading_pqc)
        return tf.keras.Model(inputs=[input_tensor], outputs=critic)

    def generate_shared_actor_critic_model(self):
        input_tensor = tf.keras.Input(shape=(len(self.actor_qubits),), dtype=tf.float32, name='input')

        # Shared ReUploadingPQC layer
        shared_re_uploading_pqc = ReUploadingPQC(self.actor_qubits, self.n_layers, self.actor_observables)([input_tensor])

        # Actor Head
        action = tf.keras.Sequential([
            Alternating(self.n_actions),
            tf.keras.layers.Lambda(lambda x: x * self.beta),
            tf.keras.layers.Softmax()
        ], name="observables-policy")(shared_re_uploading_pqc)

        # Critic Head
        critic = tf.keras.Sequential([
            Rescaling(len(self.critic_observables))
        ], name="Q-values")(shared_re_uploading_pqc)

        return tf.keras.Model(inputs=[input_tensor], outputs=[action, critic])

    def train(self):

        episode_count = 0
        running_reward = 0

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

                    # Shared Model---------------------------
                    # shared_output = self.shared_actor_critic_model(state)
                    # action_probs, critic_value = shared_output
                    #-----------------------------------------

                    action_probs = self.actor_model(state)
                    critic_value = self.critic_model(state)

                    critic_value_history.append(critic_value[0, 0])
                    action = np.random.choice(self.n_actions, p=np.squeeze(action_probs))
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

                # Calculating loss values to update our network
                history = zip(action_probs_history, critic_value_history, returns)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    diff = ret - value
                    actor_losses.append(-log_prob * diff)
                    critic_losses.append(self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

                # Shared Model----------------------------------
                total_loss = sum(actor_losses) + sum(critic_losses)
                # # Apply gradients to shared_model
                # grads = tape.gradient(total_loss, self.shared_actor_critic_model.trainable_variables)
                #
                # for optimizer, w in zip([self.shared_optimizer_in, self.shared_optimizer_var, self.shared_optimizer_out],
                #                             [self.w_in, self.w_var, self.w_out]):
                #         optimizer.apply_gradients([(grads[w], self.shared_actor_critic_model.trainable_variables[w])])
                #-----------------------------------------------

                grads = tape.gradient(total_loss, self.actor_model.trainable_variables + self.critic_model.trainable_variables)
                # actor_grads = tape.gradient(actor_losses, self.actor_model.trainable_variables)
                for optimizer, w in zip([self.actor_optimizer_in, self.actor_optimizer_var, self.actor_optimizer_out],
                                            [self.w_in, self.w_var, self.w_out]):
                        optimizer.apply_gradients([(grads[:len(self.actor_model.trainable_variables)][w], self.actor_model.trainable_variables[w])])

                # critic_grads = tape.gradient(critic_losses, self.critic_model.trainable_variables)
                for optimizer, w in zip([self.critic_optimizer_in, self.critic_optimizer_var, self.critic_optimizer_out],
                                            [self.w_in, self.w_var, self.w_out]):
                        optimizer.apply_gradients([(grads[len(self.actor_model.trainable_variables):][w], self.critic_model.trainable_variables[w])])

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

            episode_count += 1
            if episode_count % 10 == 0:
                avg_rewards = np.mean(self.episode_reward_history[-10:])
                print(
                    "Episode {}/{}, average last 10 rewards {}".format(episode_count, self.n_episodes, avg_rewards))

                if avg_rewards >= 500.0:
                    break

        self.average_timesteps_per_episode = np.mean(self.episode_length_history)
        self.episode_count = len(self.episode_reward_history)
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
                action_probs, _ = self.model(state[None, :])
                action = np.argmax(action_probs[0])
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        if render:
            self.env.close()



if __name__ == "__main__":
    actor_critic_agent = ActorCriticQuantum()
    actor_critic_agent.train()
    actor_critic_agent.play(render=True)
