import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the Q-network
def build_q_network(state_size, action_size):
    model = keras.Sequential()
    model.add(layers.Dense(24, activation='relu', input_dim=state_size))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    return model

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.99, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = build_q_network(state_size, action_size)
        self.target_model = build_q_network(state_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        target = self.model.predict(state)
        if done:
            target[0][action] = reward
        else:
            q_future = np.max(self.target_model.predict(next_state)[0])
            target[0][action] = reward + self.gamma * q_future

        with tf.GradientTape() as tape:
            q_values = self.model(state)
            loss = tf.keras.losses.mean_squared_error(target, q_values)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Create the CartPole environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create the DQN agent
agent = DQNAgent(state_size, action_size)

# Training parameters
n_episodes = 1000
max_timesteps = 500
render = False

# Main training loop
for episode in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for timestep in range(max_timesteps):
        if render:
            env.render()

        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.train(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            agent.update_target_model()
            break

    print("Episode:", episode+1, "Total Reward:", total_reward)

# Evaluate the trained agent
total_rewards = []
n_eval_episodes = 10

for _ in range(n_eval_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    episode_reward = 0

    while True:
        env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        episode_reward += reward
        state = next_state

        if done:
            break

    total_rewards.append(episode_reward)

average_reward = np.mean(total_rewards)
print("Average Reward:", average_reward)

# Close the environment
env.close()
