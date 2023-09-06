
# Update package resources to account for version changes.
import importlib, pkg_resources
importlib.reload(pkg_resources)

import tensorflow as tf
import tensorflow_quantum as tfq
import time
from datetime import datetime
import os
from typing import List, Dict, Any
from collections import defaultdict
import statistics

import matplotlib.pyplot as plt
import json
import gym, cirq, sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
from scipy.signal import savgol_filter
tf.get_logger().setLevel('ERROR')

print(tf.__version__)
print(tfq.__version__)
print(gym.__version__)
print(tf.config.list_physical_devices('GPU'))


def load_json_file(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def save_to_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, 'w') as f:
        # properly formatted
        json.dump(data, f, indent=4)

import re

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


def filter_files(directory, substring, extension):
    all_filtered_files=[]
    for item_name in os.listdir(directory):
        item_path = os.path.join(directory, item_name)
        # Check if the item is a directory and contains the word 'run' in its name
        if os.path.isdir(item_path) and 'run' in item_name:
            print([f for f in os.listdir(item_path) if substring in f and f.endswith(extension)])
            filtered_files = [f for f in os.listdir(item_path) if substring in f and f.endswith(extension)]
            sorted_filtered_files = sorted(filtered_files, key=natural_keys)
            all_filtered_files.extend(sorted_filtered_files)

    return all_filtered_files


def compute_averaged_rewards(list_of_lists: List[List[float]], method: str = 'plateau', fill_value: float = None) -> List[float]:
    """
    Takes a list of lists and returns a new list containing the average of each
    corresponding element across all the lists.

    Parameters:
        list_of_lists (List[List[float]]): List containing lists of numerical values.
        method (str): The method used to handle lists of different sizes.
                     Options are 'plateau' and 'truncate'.
        fill_value (float): If specified, fill shorter lists with this value
                            instead of plateauing.

    Returns:
        List[float]: Averaged list.
    """
    max_length = max(len(lst) for lst in list_of_lists)

    if method == 'plateau':
        # Extend each list to max_length using the last element or fill_value
        for lst in list_of_lists:
            lst += [lst[-1] if fill_value is None else fill_value] * (max_length - len(lst))

    elif method == 'truncate':
        # Truncate each list to the size of the smallest list
        min_length = min(len(lst) for lst in list_of_lists)
        list_of_lists = [lst[:min_length] for lst in list_of_lists]

    # Zip the lists to group corresponding elements together
    zipped_lists = zip(*list_of_lists)

    # Calculate the average for each group of corresponding elements
    averaged_list = [sum(values) / len(values) for values in zipped_lists]

    return averaged_list



def combine_experiment_outcomes(experiment_outcomes_list: List[Dict[str, Any]], exp_id, combining_strategy="mean", save=True) -> Dict[str, Any]:
    """
    Combine a list of experiment outcome dictionaries into a single dictionary.

    Parameters:
        experiment_outcomes_list (List[Dict[str, Any]]): List containing experiment outcome dictionaries.
        combining_strategy (str): Strategy for combining numerical values.

    Returns:
        Dict[str, Any]: Combined experiment outcome dictionary.
    """

    # Initialize a dictionary to store combined results
    combined_dict = defaultdict(lambda: defaultdict(list))

    # Iterate through each experiment outcome dictionary
    for experiment_dict in experiment_outcomes_list:
        for main_key, sub_dict in experiment_dict.items():
            if main_key in ['metrics', 'timings']:
                if isinstance(sub_dict, dict):
                    for sub_key, value in sub_dict.items():
                        # Store values in lists for later averaging or unique identification
                        combined_dict[main_key][sub_key].append(value)
                elif isinstance(sub_dict, list):
                    # Store values in lists for later averaging or unique identification
                    combined_dict[main_key][sub_dict].append(value)
                else:
                    # If it's not a dictionary, just append to list
                    combined_dict[main_key].append(sub_dict)
            else:
                combined_dict[main_key] = sub_dict

    # Convert defaultdict to dict
    combined_dict = {k: dict(v) for k, v in combined_dict.items()}

    # Average numerical metrics and keep unique config values
    for main_key, sub_dict in combined_dict.items():
        if main_key in ['metrics', 'timings']:

            if isinstance(sub_dict, dict):
                for sub_key, value_list in sub_dict.items():
                    if all(isinstance(v, (int, float)) for v in value_list):
                        if combining_strategy == "mean":
                            combined_dict[main_key][sub_key] = round(statistics.mean(value_list))
                        elif combining_strategy == "median":
                            combined_dict[main_key][sub_key] = round(statistics.median(value_list))
                    elif all(isinstance(v, list) for v in value_list):
                        max_length = max(len(v) for v in value_list)
                        # Use the plateau method to extend each list to the maximum length
                        padded_value_list = [v + [v[-1]] * (max_length - len(v)) if v else [0] * max_length for v in
                                             value_list]
                        if combining_strategy == "mean":
                            combined_dict[main_key][sub_key] = [round(statistics.mean(vals)) for vals in zip(*padded_value_list)]
                        elif combining_strategy == "median":
                            combined_dict[main_key][sub_key] = [round(statistics.median(vals)) for vals in zip(*padded_value_list)]
                    else:
                        combined_dict[main_key][sub_key] = list(set(value_list))


    if save:
        file_name = "combined_experiment_outcomes_{}_{}.json".format(combining_strategy, datetime.now().strftime("%Y%m%d-%H%M%S"))
        save_to_json(combined_dict, os.path.join(exp_id, file_name))
        print(f"Combined experiment outcomes saved to {file_name}")

    return combined_dict


def moving_average(data, window_size):
    """Apply a simple moving average."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def smooth_curve_MA(y_values, smoothing_level=20, save=False):
    """
    Smooth a curve using a moving average.

    Parameters:
    - y_values (list): List of y-values.
    - smoothing_level (int): Window size for the moving average.
    - save (bool): Whether to save the smoothed data to a JSON file.
    """
    # Perform smoothing using moving average
    y_smooth = moving_average(y_values, smoothing_level).tolist()
    # Save smoothed data points to JSON if required
    if save:
        data_to_save = {'x': x.tolist(), 'y_smooth': y_smooth.tolist()}
        save_to_json(data_to_save, filename)
        print(f"Smoothed data saved to {filename}")

    return smoothed_values

def smooth_curve(y, exp_id, smoothing_method="EMA", smoothing_level=0.9, save=True, save_file_name="smoothed_data_{}.json".format(datetime.now().strftime("%Y%m%d-%H%M%S"))):
    """
    Smooths a curve based on the specified smoothing method and level.

    Parameters:
    - data_dict (dict): Dictionary containing "rewards" as a list of floats.
    - smoothing_method (str): Method for smoothing ('EMA' or 'Savitzky-Golay Filter').
    - smoothing_level (int/float): Level of smoothing.
    - save (bool): Whether to save smoothed data to JSON.
    - save_file_name (str): File name to save the JSON.

    Returns:
    - list: Smoothed rewards
    """

    if smoothing_method == 'EMA':
        # Smoothing by Exponential Moving Average (EMA)
        if smoothing_level <= 0 or smoothing_level >= 1:
            raise ValueError("For EMA, smoothing level should be in (0, 1)")

        y_smooth = []
        for point in y:
            if y_smooth:
                previous = y_smooth[-1]
                y_smooth.append(previous * smoothing_level + point * (1 - smoothing_level))
            else:
                y_smooth.append(point)

    elif smoothing_method == 'SG_filter':
        # Smoothing by Savitzky-Golay Filter
        if smoothing_level % 2 == 0 or smoothing_level < 3: # Smoothing level must be odd and >= 3 -
            raise ValueError("For Savitzky-Golay, smoothing level must be an odd integer >= 3")

        y_smooth = savgol_filter(y, smoothing_level, 2).tolist()

    else:
        raise ValueError("Invalid smoothing method. Choose 'EMA' or 'SG_filter'.")

    # Save the smoothed data to a JSON file if required
    if save:
        data_to_save = {'x': list(range(len(y))), 'y': list(y), 'x_smooth': list(range(len(y_smooth))), 'y_smooth': y_smooth}
        save_to_json(data_to_save, os.path.join(exp_id, save_file_name))
        print(f"Smoothed data saved to {save_file_name}")

    return y_smooth

def adjust_text_size(basic_size):
    """
    Function to adjust the text sizes for the plot based on a basic text size.
    Returns a dictionary containing the updated text sizes.
    """
    return {
        "title_size": basic_size + 6,
        "label_size": basic_size + 2,
        "tick_size": basic_size,
        "legend_size": basic_size + 1,
    }


def plot_smoothed_rewards(episode_rewards, exp_id, smoothing_method="EMA", smoothing_level=0.9, save=True,
                          prefix_str="", save_plot_name=None, fig_size=(10, 6),
                          plot_title="Rewards over Episodes"):
    """Plot rewards over episodes with an option for smoothed curve."""

    smoothed_rewards = smooth_curve(episode_rewards, exp_id, smoothing_method=smoothing_method, smoothing_level=smoothing_level,
                                    save=False, save_file_name="smoothed_rewards_{}-{}_{}.json".format(smoothing_method,
                                    smoothing_level, datetime.now().strftime("%Y%m%d-%H%M%S")))

    basic_text_size = 12
    line_width = 2
    # Adjust text sizes based on the provided basic_text_size
    text_sizes = adjust_text_size(basic_text_size)

    plt.figure(figsize=fig_size)
    plt.plot(episode_rewards, label="Original", alpha=0.5, linewidth=line_width)
    plt.plot(smoothed_rewards, label=f"Smoothed (factor={smoothing_level})", linewidth=line_width)
    plt.xlabel('Episodes', fontsize=text_sizes["label_size"])
    plt.ylabel('Reward', fontsize=text_sizes["label_size"])
    plt.title(plot_title, fontsize=text_sizes["title_size"])

    plt.legend(fontsize=text_sizes["legend_size"])

    # Adjust tick sizes
    plt.xticks(fontsize=text_sizes["tick_size"])
    plt.yticks(fontsize=text_sizes["tick_size"])

    if save_plot_name:
        plot_save_path = os.path.join(exp_id, save_plot_name)
    else:
        plot_save_path = os.path.join(exp_id, "{}_smoothed_rewards_{}-{}_{}.png".format(prefix_str, smoothing_method,
                                        smoothing_level, datetime.now().strftime("%Y%m%d-%H%M%S")))
    if save:
        plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_averaged_rewards(exp_reward_histories, exp_id, smoothing_method="EMA", smoothing_level=0.9,
                          save=True, fig_size=(10, 8), plot_title="Rewards over Episodes"):

    basic_text_size = 12
    line_width = 2
    # Adjust text sizes based on the provided basic_text_size
    text_sizes = adjust_text_size(basic_text_size)
    plt.figure(figsize=fig_size)

    max_length = max(len(lst) for lst in exp_reward_histories)
    n_exps = len(exp_reward_histories)

    # Initialize an array to store the sum of rewards at each episode across all reruns
    sum_rewards = np.zeros(max_length)

    # Loop through each rerun
    for i, exp_reward_history in enumerate(exp_reward_histories):
        # Use the plateau method to fill up shorter lists
        episode_rewards = np.pad(exp_reward_history, (0, max_length - len(exp_reward_history)), 'edge')

        # Add the rewards from this rerun to sum_rewards
        sum_rewards += episode_rewards

        # Plot individual reward history with alpha=0.4
        plt.plot(episode_rewards, alpha=0.4, label=f'Run {i + 1}', linewidth=1)

    # Calculate the average rewards
    avg_rewards = sum_rewards / n_exps

    smoothed_avg_rewards = smooth_curve(avg_rewards, exp_id, smoothing_method=smoothing_method,
                                        smoothing_level=smoothing_level, save=save,
                                        save_file_name="{}-{}_averaged_smoothed_rewards_{}.json".format
                                        (smoothing_method, smoothing_level,
                                        datetime.now().strftime("%Y%m%d-%H%M%S")))

    # Plot the average reward curve on top
    plt.plot(avg_rewards, linewidth=2, color='black', label='Average')
    # Plot the smoothed average reward curve on top
    plt.plot(smoothed_avg_rewards, linewidth=2, color='orange', label='Smoothed Average')

    plt.xlabel('Episodes', fontsize=text_sizes["label_size"])
    plt.ylabel('Reward', fontsize=text_sizes["label_size"])
    plt.title(plot_title, fontsize=text_sizes["title_size"])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=text_sizes["legend_size"])

    # plt.tight_layout(rect=[0, 0.1, 1, 1])  # Make room for the legend below
    # Adjust layout to keep plot size constant while allowing space for legend
    plt.subplots_adjust(bottom=0.2)  # Make room at the bottom
    plt.tight_layout()

    # Adjust tick sizes
    plt.xticks(fontsize=text_sizes["tick_size"])
    plt.yticks(fontsize=text_sizes["tick_size"])

    plot_save_file_name = "averaged_rewards_{}-{}_{}.png".format(smoothing_method, smoothing_level,
                                                                    datetime.now().strftime("%Y%m%d-%H%M%S"))
    if save:
        plt.savefig(os.path.join(exp_id, plot_save_file_name), bbox_inches='tight', pad_inches=0.1)
    plt.show()




def plot_timesteps_over_episodes(json_data, smoothing_factor=0.9, save=True):
    """Plot timesteps per episode over episodes with an option for smoothed curve."""
    episode_timesteps = json_data['metrics']['episode_lengths']
    smoothed_timesteps = smooth_curve(episode_timesteps, smoothing_factor)

    plt.figure()
    plt.plot(episode_timesteps, label="Original", alpha=0.5)
    plt.plot(smoothed_timesteps, label=f"Smoothed (factor={smoothing_factor})")
    plt.xlabel('Episodes')
    plt.ylabel('Timesteps')
    plt.title('Timesteps over Episodes')
    plt.legend()

    if save:
        plt.savefig('timesteps_over_episodes.png')
    else:
        plt.show()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]


def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

def generate_circuit(qubits, n_layers):
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

class Rescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Rescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply((inputs+1)/2, tf.repeat(self.w,repeats=tf.shape(inputs)[0],axis=0))



class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])


class Alternating(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(Alternating, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.constant([[(-1.)**i for i in range(output_dim)]]), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.matmul(inputs, self.w)



def A2C_old_train(self):
    env = gym.make(self.env_name)
    num_episodes = 200

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        with tf.GradientTape() as tape:
            for timestep in range(1, 10000): # Run for a max to avoid infinite loop
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)

                # Predict action probabilities and estimated future rewards from environment state
                action_probs, values = self.actor_model(state_tensor), self.critic_model(state_tensor)
                action = np.random.choice(self.n_actions, p=np.squeeze(action_probs))

                # Take action in the environment and compute the reward and next state
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                # Record reward and next state in memory for learning
                self.rewards_history.append(reward)
                self.state_value_history.append(values[0, 0])

                if done:
                    break

                state = next_state

        # Compute expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are used to compute the loss values
        returns = []
        discounted_sum = 0
        for r in self.rewards_history[::-1]:
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Compute loss values to update our network
        actor_losses = []
        critic_losses = []
        for prob, value, ret in zip(action_probs, self.state_value_history, returns):
            # Compute the loss for the critic
            critic_losses.append(self.huber_loss(tf.expand_dims(value, 0), ret))

            # Compute the loss for the actor
            log_prob = tf.math.log(prob)
            advantage = ret - value
            actor_losses.append(-log_prob * advantage)

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Clear the loss and reward history
        self.rewards_history.clear()
        self.state_value_history.clear()

        # Log details
        self.episode_reward_history.append(episode_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {episode_reward}")

    self.plot_rewards()


