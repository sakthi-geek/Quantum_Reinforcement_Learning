
# Importing required libraries
import gym
import numpy as np
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import random


class DeepQLearningClassical:
    def __init__(self, n_actions=2, gamma=0.99, n_episodes=2000, batch_size=16):
        # Hyperparameters
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_episodes = n_episodes

        # Initialize models
        self.model = self.generate_model_Qlearning()
        self.model_target = self.generate_model_Qlearning()
        self.model_target.set_weights(self.model.get_weights())

        tf.keras.utils.plot_model(self.model, show_shapes=True, dpi=70)
        tf.keras.utils.plot_model(self.model_target, show_shapes=True, dpi=70)

        # Initialize replay memory and other variables
        self.max_memory_length = 10000
        self.replay_memory = deque(maxlen=self.max_memory_length)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter - range - 0 to 1 - 1 means decay is not happening - 0 means decay is happening very fast
        self.steps_per_update = 10
        self.steps_per_target_update = 30

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Initialize environment
        self.env = gym.make("CartPole-v1")

        self.episode_reward_history = []

    def generate_model_Qlearning(self):
        """Generates a Keras model for the Q-function approximator."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),
            # tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.n_actions)
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def interact_env(self, state):
        """Performs an interaction step in the environment."""
        state_array = np.array(state)
        state = state_array.reshape(1, -1)

        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])

        next_state, reward, done, _ = self.env.step(action)

        interaction = {'state': state_array, 'action': action, 'next_state': next_state.copy(),
                       'reward': reward, 'done': np.float32(done)}

        return interaction

    def sample_interaction(self, state):
        """Sample an interaction with the environment."""
        interaction = self.interact_env(state)
        self.replay_memory.append(interaction)
        return interaction['next_state'], interaction['reward'], interaction['done']

    def update_models(self, step_count):
        """Update models at appropriate intervals."""
        if step_count % self.steps_per_update == 0:
            if len(self.replay_memory) < self.batch_size:
                return
            batch = random.sample(self.replay_memory, self.batch_size)

            states = np.array([x['state'] for x in batch])
            actions = np.array([x['action'] for x in batch])
            rewards = np.array([x['reward'] for x in batch], dtype=np.float32)
            next_states = np.array([x['next_state'] for x in batch])
            done_flags = np.array([x['done'] for x in batch], dtype=np.float32)

            self.update_Q_model(states, actions, rewards, next_states,
                                done_flags)  # Pass the pre-converted numpy arrays here

        if step_count % self.steps_per_target_update == 0:
            self.model_target.set_weights(self.model.get_weights())

    @tf.function
    def update_Q_model(self, states, actions, rewards, next_states, done_flags):
        """Perform a Q-learning update using a batch of training data."""
        target = rewards + (1 - done_flags) * self.gamma * tf.reduce_max(self.model_target(next_states), axis=1)
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            one_hot_actions = tf.one_hot(actions, depth=self.n_actions, dtype=tf.float32)
            predicted_q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            loss = tf.reduce_mean(tf.square(target - predicted_q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

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

                # Update models
                self.update_models(step_count)

                if done:
                    break

            self.epsilon = max(self.epsilon * self.decay_epsilon, self.epsilon_min)

            self.episode_reward_history.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_rewards = np.mean(self.episode_reward_history[-10:])
                print(f"Episode {episode + 1}/{self.n_episodes}, average last 10 episode rewards: {avg_rewards}")

                if avg_rewards >= 500.0:
                    break
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



###----------------------------------------------
# import gym
# import numpy as np
# import tensorflow as tf
# from collections import deque
# import random
#
#
# # Helper functions can go in a separate helper.py file
# # Here's a helper function for the Q-network
# def build_model(state_shape, n_actions):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(24, activation='relu', input_shape=state_shape),
#         tf.keras.layers.Dense(24, activation='relu'),
#         tf.keras.layers.Dense(n_actions, activation='linear')
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
#     return model
#
#
# class DQNAgent:
#     def __init__(self, state_shape, n_actions):
#         self.state_shape = state_shape  # Shape of the state
#         self.n_actions = n_actions  # Number of actions
#         self.memory = deque(maxlen=10000)  # Replay memory
#
#         # Q-network and target network
#         self.model = build_model(state_shape, n_actions)
#         self.target_model = build_model(state_shape, n_actions)
#         self.update_target_model()
#
#         self.gamma = 0.99  # Discount factor
#         self.epsilon = 1.0  # Exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.99
#
#     def update_target_model(self):
#         # Copy weights to target model
#         self.target_model.set_weights(self.model.get_weights())
#
#     def remember(self, state, action, reward, next_state, done):
#         # Store experience in replay memory
#         self.memory.append((state, action, reward, next_state, done))
#
#     def act(self, state):
#         # Epsilon-greedy action selection
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.n_actions)
#         q_values = self.model.predict(state)
#         return np.argmax(q_values[0])
#
#     def replay(self, batch_size):
#         # Train the Q-network using experience replay
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target += self.gamma * np.amax(self.target_model.predict(next_state)[0])
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#
#     def load(self, name):
#         self.model.load_weights(name)
#
#     def save(self, name):
#         self.model.save_weights(name)
#
#
# # Initialize environment and the agent
# env = gym.make('CartPole-v1')
# state_shape = (env.observation_space.shape[0],)
# n_actions = env.action_space.n
#
# agent = DQNAgent(state_shape, n_actions)
#
# # Training the agent
# EPISODES = 1000
# for e in range(1, EPISODES + 1):
#     state = env.reset()
#     state = np.reshape(state, [1, state_shape[0]])
#     for time in range(500):
#         # env.render()
#         action = agent.act(state)
#         next_state, reward, done, _ = env.step(action)
#         next_state = np.reshape(next_state, [1, state_shape[0]])
#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             agent.update_target_model()
#             print(f"Episode: {e}/{EPISODES}, Score: {time}, Epsilon: {agent.epsilon:.2}")
#             break
#     if len(agent.memory) > 32:
#         agent.replay(32)
#
# # Close the environment
# env.close()