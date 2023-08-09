import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from collections import defaultdict


class ActorCriticClassical:
    def __init__(self, gamma=0.99, n_episodes=2000, max_steps_per_episode=10000, learning_rate=0.01):
        # Configuration parameters
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = learning_rate

        self.env = gym.make("CartPole-v1")
        num_inputs = 4
        num_actions = 2
        num_hidden = 128

        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        action = layers.Dense(num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        self.model = keras.Model(inputs=inputs, outputs=[action, critic])

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.huber_loss = keras.losses.Huber()
        self.episode_reward_history = []

        # Environment metrics
        self.pole_angle_history = []
        self.pole_angular_velocity_history = []
        self.cart_position_history = []
        self.cart_velocity_history = []
        self.action_distribution = defaultdict(int)
        self.termination_reasons = {"pole_angle": 0, "cart_position": 0}

        # General metrics
        self.episode_lengths = []
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []
        self.advantage_values_history = []
        self.gradient_magnitudes_history = []
        self.episode_rewards_min = []
        self.episode_rewards_max = []
        self.episode_rewards_mean = []
        self.episode_rewards_std = []

        # Early stopping parameters
        self.patience = 500  # Number of episodes to wait for improvement
        self.min_improvement = 5  # Minimum improvement in average reward to be considered actual improvement
        self.best_avg_reward = -np.inf  # Best average reward observed so far
        self.waited_episodes = 0  # Number of episodes waited since the last improvement

    def train(self):
        episode_count = 0
        running_reward = 0
        N = 10  # You can adjust this to capture metrics every N episodes
        best_weights = None  # Store the best model weights

        for _ in range(self.n_episodes):
            state = self.env.reset()
            action_probs_history = []
            critic_value_history = []
            rewards_history = []
            episode_reward = 0
            episode_entropy = []
            episode_advantages = []

            with tf.GradientTape() as tape:
                for timestep in range(1, self.max_steps_per_episode):
                    state = tf.convert_to_tensor(state)
                    state = tf.expand_dims(state, 0)
                    action_probs, critic_value = self.model(state)
                    critic_value_history.append(critic_value[0, 0])
                    action = np.random.choice(self.env.action_space.n, p=np.squeeze(action_probs))
                    action_probs_history.append(tf.math.log(action_probs[0, action]))
                    state, reward, done, _ = self.env.step(action)

                    rewards_history.append(reward)
                    episode_reward += reward

                    # Update environment-specific metrics - GRANULAR
                    if episode_count % N == 0:
                        pole_angle, pole_velocity, cart_position, cart_velocity = state
                        self.pole_angle_history.append(pole_angle)
                        self.pole_angular_velocity_history.append(pole_velocity)
                        self.cart_position_history.append(cart_position)
                        self.cart_velocity_history.append(cart_velocity)
                        self.action_distribution[action] += 1

                        # Entropy of the policy
                        entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs))
                        episode_entropy.append(entropy.numpy())

                    # Update termination reasons (only if done)
                    if done:
                        # Store the episode length
                        self.episode_lengths.append(timestep)
                        if abs(pole_angle) > 0.2094:  # ~12 degrees
                            self.termination_reasons["pole_angle"] += 1
                        elif abs(cart_position) > 2.4:
                            self.termination_reasons["cart_position"] += 1
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
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Storing actor and critic losses separately
                self.actor_loss_history.append(np.mean(actor_losses))
                self.critic_loss_history.append(np.mean(critic_losses))

                # Calculate average entropy and advantage values and store them
                self.entropy_history.append(np.mean(episode_entropy))

                # Storing advantage values
                advantages = [ret - value for ret, value in zip(returns, critic_value_history)]
                self.advantage_values_history.append(np.mean(advantages))

                # Gradient magnitudes
                gradient_magnitudes = [tf.norm(grad).numpy() for grad in grads if grad is not None]
                self.gradient_magnitudes_history.append(np.mean(gradient_magnitudes))

            # Storing rewards stats - GRANULAR
            if episode_count % N == 0:
                self.episode_rewards_min.append(np.min(rewards_history))
                self.episode_rewards_max.append(np.max(rewards_history))
                self.episode_rewards_mean.append(np.mean(rewards_history))
                self.episode_rewards_std.append(np.std(rewards_history))

            # Early stopping check
            if len(self.episode_reward_history) >= self.patience:
                avg_last_rewards = np.mean(self.episode_reward_history[-self.patience:])
                if avg_last_rewards - self.best_avg_reward >= self.min_improvement:
                    self.best_avg_reward = avg_last_rewards
                    self.waited_episodes = 0
                    best_weights = self.model.get_weights()  # Save the best weights
                else:
                    self.waited_episodes += 1
                    if self.waited_episodes >= self.patience:
                        print(f"Early stopping after {self.waited_episodes} episodes without improvement.")
                        self.model.set_weights(best_weights)  # Restore the best weights
                        break

            episode_count += 1
            if episode_count % 10 == 0:
                avg_rewards = np.mean(self.episode_reward_history[-10:])
                print("Episode {}/{}, average last 10 rewards {}".format(episode_count, self.n_episodes, avg_rewards))

                if avg_rewards >= 500.0:
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
                # state = tf.convert_to_tensor(state)
                # state = tf.expand_dims(state, 0)
                action_probs, _ = self.model(state[None, :])
                action = np.argmax(action_probs[0])
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        if render:
            self.env.close()


if __name__ == "__main__":
    actor_critic_agent = ActorCriticClassical()
    actor_critic_agent.train()
    actor_critic_agent.play(render=True)
