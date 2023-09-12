import tensorflow as tf
import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

class REINFORCEAgent:
    def __init__(self, env_name='CartPole-v1', seed=39, n_inputs=4, n_hidden=[32], n_actions=2, gamma=0.99,
                 batch_size=16, n_episodes=2000, learning_rate=0.01):

        # Environment -----------------------
        self.seed = seed
        self.env_name = env_name
        self.env = gym.make(env_name)

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        print(self.state_size, self.action_size)

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        # random.seed(self.seed)

        # Configuration parameters----------------------
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.state_bounds = np.array([2.4, 2.5, 0.21, 2.5])

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_actions = n_actions

        # Optimizer with Learning Rate Scheduler
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=10000,  # decay the learning rate after every 10000 steps
            decay_rate=0.99, )  # decay rate factor
        # Model ---------------------------------------
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=True)
        self.model = self.build_model(n_inputs=self.n_inputs, n_hidden=self.n_hidden, n_actions=self.n_actions)

        self.input_shape = self.model.input_shape
        self.output_shape = self.model.output_shape
        self.trainable_params = self.model.count_params()

        self.config_params = None
        # print all parameters
        print("----------------------")
        print("Environment: ", self.env_name)
        print("Seed: ", self.seed)
        print("Gamma: ", self.gamma)
        print("Number of Episodes: ", self.n_episodes)
        print("Batch size: ", self.batch_size)
        print("Learning Rate: ", self.learning_rate)
        print("State Bounds: ", self.state_bounds)
        print("Inputs: ", self.n_inputs)
        print("hidden Layers: ", self.n_hidden)
        print("Number of Actions: ", self.n_actions)
        print("Model: ", self.model.summary())
        print("Input Shape: ", self.input_shape)
        print("Output Shape: ", self.output_shape)
        print("Trainable Parameters: ", self.trainable_params)
        print("lr_schedule: ", {"initial_learning_rate": lr_schedule.initial_learning_rate,
                                "decay_steps": lr_schedule.decay_steps,
                                "decay_rate": lr_schedule.decay_rate})
        print("----------------------")

        # Metrics ---------------------------
        self.episode_reward_history = []
        self.episode_length_history = []



    def build_model(self, n_inputs=4, n_hidden=[32], n_actions=2):
        """Builds a Keras model for the policy."""
        model = tf.keras.models.Sequential()
        for i, n_nodes in enumerate(n_hidden):
            if i == 0:
                model.add(layers.Dense(n_nodes, input_dim=n_inputs, activation='relu'))
            else:
                model.add(layers.Dense(n_nodes, activation='relu'))
        model.add(layers.Dense(n_actions, activation='softmax'))
        return model

    def gather_episodes(self):
        """Gathers `batch_size` episodes."""
        trajectories = [defaultdict(list) for _ in range(self.batch_size)]
        envs = [gym.make(self.env_name) for _ in range(self.batch_size)]
        # Set seed for each environment instance for reproducibility
        seeded_envs = []
        if self.seed is not None:
            for i, env in enumerate(envs):
                env.seed(self.seed + i)
                seeded_envs.append(env)

        done = [False for _ in range(self.batch_size)]
        states = [e.reset()[0] for e in seeded_envs]

        while not all(done):
            unfinished_ids = [i for i in range(self.batch_size) if not done[i]]
            normalized_states = [s / self.state_bounds for i, s in enumerate(states) if not done[i]]

            for i, state in zip(unfinished_ids, normalized_states):
                trajectories[i]['states'].append(state)

            states = tf.convert_to_tensor(normalized_states)
            action_probs = self.model([states])

            states = [None for i in range(self.batch_size)]
            for i, policy in zip(unfinished_ids, action_probs.numpy()):
                action = np.random.choice(self.action_size, p=policy)
                states[i], reward, done[i], _ = envs[i].step(action)
                trajectories[i]['actions'].append(action)
                trajectories[i]['rewards'].append(reward)

        return trajectories

    @staticmethod
    def compute_returns(rewards_history, gamma):
        """Compute discounted returns with discount factor `gamma`."""
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = returns.tolist()

        return returns

    @tf.function
    def reinforce_update(self, states, actions, returns):
        """Updates the policy with a reinforce update."""
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            logits = self.model(states)
            p_actions = tf.gather_nd(logits, actions)
            log_probs = tf.math.log(p_actions)
            loss = tf.math.reduce_sum(-log_probs * returns) / self.batch_size
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train(self):
        """Train the agent."""
        # required_consistency = 5  # The number of times the agent needs to last 500 timesteps to stop training

        for batch in range(self.n_episodes // self.batch_size):
            episodes = self.gather_episodes()

            states = np.concatenate([ep['states'] for ep in episodes])
            actions = np.concatenate([ep['actions'] for ep in episodes])
            rewards = [ep['rewards'] for ep in episodes]
            returns = np.concatenate([self.compute_returns(ep_rwds, self.gamma) for ep_rwds in rewards])
            returns = np.array(returns, dtype=np.float32)

            id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

            self.reinforce_update(states, id_action_pairs, returns)

            for ep_rwds in rewards:
                self.episode_reward_history.append(np.sum(ep_rwds))
                self.episode_length_history.append(len(ep_rwds))

            avg_rewards = np.mean(self.episode_reward_history[-10:])

            print('Finished episode', (batch + 1) * self.batch_size, 'Average rewards: ', avg_rewards)

            if avg_rewards >= 500.0:
                break

            # # Check if the agent lasted 500 timesteps for the last 'required_consistency' episodes
            # if len(self.episode_length_history) >= required_consistency:
            #     last_n_episodes = self.episode_length_history[-required_consistency:]
            #     if all(length >= 500 for length in last_n_episodes):
            #         print("Training stopped: Agent lasted 500 timesteps consistently for {} episodes.".format(
            #             required_consistency))
            #         break

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
        # Update the target model as well
        self.model_target.set_weights(self.model.get_weights())

    def plot_rewards(self):
        """Plot the episode reward history of the agent."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_reward_history)
        plt.xlabel('Epsiode')
        plt.ylabel('Collected rewards')
        plt.show()

    def play(self):
        """Play an episode using the trained policy."""
        env = gym.make(self.env_name)
        observation, info = env.reset()

        for _ in range(500):
            states = tf.convert_to_tensor(observation.reshape(1, -1) / self.state_bounds)
            action_probs = np.asarray(self.model(states)).astype('float64')  # issue with float32
            action_probs = action_probs / np.sum(action_probs)
            action = np.random.choice(self.action_size, p=action_probs[0])
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()


if __name__ == "__main__":
    agent = REINFORCEAgent()
    agent.train()

