# %%
import tensorflow as tf
import gymnasium as gym
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# %%
#defining a standard dense NN

def build_model(state_size, action_size, learning_rate = 0.001):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(16, input_dim = state_size, activation = 'relu'))
    model.add(tf.keras.layers.Dense(16))
    model.add(tf.keras.layers.Dense(2, activation = 'relu', kernel_initializer='RandomNormal'))
    model.add(tf.keras.layers.Dense(action_size, activation = 'softmax'))
    return model

# %%
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = build_model(state_size, action_size, 0.1)
#print(len(model.layers))
model.summary()

# %%
#function that gathers episodes of interaction with the environment
def gather_episodes(state_bounds, n_actions, model, n_episodes, env_name):
    """Interact with environment in batched fashion."""

    trajectories = [defaultdict(list) for _ in range(n_episodes)]
    envs = [gym.make(env_name) for _ in range(n_episodes)]
    

    done = [False for _ in range(n_episodes)]
    states = [e.reset()[0] for e in envs]

    while not all(done):
        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        normalized_states = [s/state_bounds for i, s in enumerate(states) if not done[i]]

        for i, state in zip(unfinished_ids, normalized_states):
            trajectories[i]['states'].append(state)

        # Compute policy for all unfinished envs in parallel
        states = tf.convert_to_tensor(normalized_states)
        action_probs = model([states])

        # Store action and transition all environments to the next state
        states = [None for i in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs.numpy()):
            #print(policy)
            action = np.random.choice(n_actions, p=policy)
            states[i], reward, done[i], _, _ = envs[i].step(action)
            trajectories[i]['actions'].append(action)
            trajectories[i]['rewards'].append(reward)

    return trajectories

# %%
def compute_returns(rewards_history, gamma):
    """Compute discounted returns with discount factor `gamma`."""
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize them for faster and more stable learning
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    returns = returns.tolist()

    return returns

# %%
state_bounds = np.array([2.4, 2.5, 0.21, 2.5])
n_actions = 2
gamma = 0.99
batch_size = 16
n_episodes = 2000

optimizer = tf.keras.optimizers.Adam(learning_rate=0.02, amsgrad=True)

# Assign the model parameters to each optimizer
w_in, w_var, w_out = 1, 0, 2

# %%
#function that updates the policy using states, actions and returns
@tf.function
def reinforce_update(states, actions, returns, model):
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    returns = tf.convert_to_tensor(returns)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        logits = model(states)
        p_actions = tf.gather_nd(logits, actions)
        log_probs = tf.math.log(p_actions)
        loss = tf.math.reduce_sum(-log_probs * returns) / batch_size
    grads = tape.gradient(loss, model.trainable_variables)
    #for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
    #    optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# %%
#main training loop of the agent
env_name = "CartPole-v1"

# Start training the agent
episode_reward_history = []
for batch in range(n_episodes // batch_size):
    # Gather episodes
    episodes = gather_episodes(state_bounds, n_actions, model, batch_size, env_name)

    # Group states, actions and returns in numpy arrays
    states = np.concatenate([ep['states'] for ep in episodes])
    actions = np.concatenate([ep['actions'] for ep in episodes])
    rewards = [ep['rewards'] for ep in episodes]
    returns = np.concatenate([compute_returns(ep_rwds, gamma) for ep_rwds in rewards])
    returns = np.array(returns, dtype=np.float32)

    id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

    # Update model parameters.
    reinforce_update(states, id_action_pairs, returns, model)

    # Store collected rewards
    for ep_rwds in rewards:
        episode_reward_history.append(np.sum(ep_rwds))

    avg_rewards = np.mean(episode_reward_history[-10:])

    print('Finished episode', (batch + 1) * batch_size,
          'Average rewards: ', avg_rewards)

    if avg_rewards >= 500.0:
        break

plt.figure(figsize=(10,5))
plt.plot(episode_reward_history)
plt.xlabel('Epsiode')
plt.ylabel('Collected rewards')
plt.show()


# %%
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(500):
    states = tf.convert_to_tensor(observation.reshape(1, -1) / state_bounds)
    action_probs = np.asarray(model(states)).astype('float64')  # issue with float32
    action_probs = action_probs/np.sum(action_probs)
    action = np.random.choice(n_actions, p=action_probs[0])
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
# %%
