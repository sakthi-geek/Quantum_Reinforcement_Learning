
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
import re
from scipy import stats
import pandas as pd
from scipy.stats import ttest_ind, f, f_oneway, levene
from scipy import stats
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from statsmodels.stats.power import FTestAnovaPower
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower
import glob
import math

import matplotlib.pyplot as plt
import json, csv
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

def confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for given data.
    """
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    z_value = stats.norm.ppf((1 + confidence) / 2)
    margin_error = z_value * (std_dev / np.sqrt(n))
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    return lower_bound, upper_bound

def standard_deviation_of_mean(std_dev, n):
    """
    Calculate the Standard Deviation of the Mean (SDOM)
    """
    return std_dev / np.sqrt(n)

def coefficient_of_variation(mean, std_dev):
    """
    Calculate the Coefficient of Variation (CV)
    """
    return (std_dev / mean) * 100

def perform_t_test(rewards_1, rewards_2):
    """
    Performs an independent t-test between rewards_1 and rewards_2.
    Returns the t statistic and p-value.
    """
    return ttest_ind(rewards_1, rewards_2, equal_var=False)

def save_t_test_results_to_csv(results_dict, csv_filename):
    """
    Saves t-test results to a CSV file.
    """
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(csv_filename, index=False)

def cohen_d(x, y):
    """Compute Cohen's d effect size between two groups x, and y."""
    nx = len(x)
    ny = len(y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2))
    return (mean_x - mean_y) / pooled_std


def welchs_anova(*groups):
    # Number of groups
    k = len(groups)

    # Sample sizes, means, and variances for each group
    n = np.array([len(g) for g in groups])
    means = np.array([np.mean(g) for g in groups])
    variances = np.array([np.var(g, ddof=1) for g in groups])

    # Grand mean and pooled variance
    grand_mean = np.sum(n * means) / np.sum(n)
    pooled_var = np.sum((n - 1) * variances) / np.sum(n - 1)

    # Welch's ANOVA F-statistic
    numerator = np.sum((n * (means - grand_mean) ** 2) / (variances + pooled_var))
    denominator = np.sum((n - 1) * variances / (variances + pooled_var))

    f_statistic = numerator / denominator

    # Degrees of freedom for Welch's ANOVA
    dfn = k - 1
    dfd = (3 * (k - 1) * np.sum(1 / (n - 1))) / (np.sum(1 / (n - 1)) ** 2 - np.sum(1 / (n - 1) ** 2))

    # p-value
    p_value = 1 - f.cdf(f_statistic, dfn, dfd)

    return f_statistic, p_value

def deep_dict_compare(d1, d2, ignored_keys):
    """Recursive function to deeply compare two dictionaries while ignoring specified keys."""
    # Compare keys
    keys1, keys2 = set(d1.keys()), set(d2.keys())
    for key in ignored_keys:
        keys1.discard(key)
        keys2.discard(key)

    if keys1 != keys2:
        return False

    # Compare values
    for key in keys1:
        val1, val2 = d1[key], d2[key]
        if isinstance(val1, dict) and isinstance(val2, dict):
            if not deep_dict_compare(val1, val2, ignored_keys):
                return False
        elif val1 != val2:
            return False

    return True


def update_json_files(filepaths, save=False):
    for filepath in filepaths:
        # Read original JSON
        with open(filepath, 'r') as f:
            original_json = json.load(f)

        # Modify JSON data
        updated_json = original_json.copy()
        updated_json['metrics']['episode_length_history'] = updated_json['metrics']['episode_reward_history']
        total_episodes = updated_json['metrics']['episode_count']
        total_timesteps = sum(updated_json['metrics']['episode_length_history'])
        updated_json['metrics'][
            'average_timesteps_per_episode'] = total_timesteps / total_episodes if total_episodes else 0

        # Verify both JSONs are identical except for the two fields we changed
        if not deep_dict_compare(original_json, updated_json,
                                 {'episode_length_history', 'average_timesteps_per_episode'}):
            print(f"Something went wrong with {filepath}, JSONs are not identical!")
            continue

        # Save updated JSON
        if save:
            updated_filepath = filepath.split('.')[0] + '_updated.json'
            with open(updated_filepath, 'w') as f:
                json.dump(updated_json, f, indent=4)

        print(f"Updated {filepath}")

def calculate_agg_metrics_and_advanced_stats(select_experiment_dicts, experiment_type, rl_agent, experiment_names, save=False):
    """
    Perform comprehensive comparative metrics, uncertainties, and statistical tests on selected experiments.

    Parameters:
    all_experiment_dicts: list of dict
        List of dictionaries, each containing the results of an experiment.
    experiment_type: str
        Type of experiments to compare: "Classical vs Quantum" or "Quantum N layer"
    rl_agent: str
        Reinforcement Learning agent type: "Policy_Gradient_REINFORCE", "Deep_Q_Learning", "Actor_Critic"
    experiment_names: dict with exp_ids as keys and exp_names as values

    Keyword Arguments:
    save: bool, optional

    Returns:
        Whether to save the output DataFrame to a CSV file.

    Returns:
    None
    """

    # Create empty DataFrames to store results
    advanced_stats_df = pd.DataFrame()
    agg_metrics = defaultdict(list)
    headers = []

    for exp_dict in select_experiment_dicts:
        exp_id = exp_dict['experiment_config']['experiment_id']
        metrics = exp_dict['metrics']

        headers.append(exp_id)
        agg_metrics["Experiment_Name"].append(experiment_names[exp_id])
        agg_metrics["N_Episodes"].append(int(np.mean(metrics['episode_count'])))
        agg_metrics["Training_Time"].append(metrics['training_time'])
        agg_metrics["Training_Time_per_Episode"].append(metrics['training_time'] / np.mean(metrics['episode_count']))
        agg_metrics["Total_Reward"].append(np.sum(metrics['episode_reward_history']))
        agg_metrics["Efficiency"].append(np.sum(metrics['episode_reward_history']) / metrics['training_time'])
        agg_metrics["Mean_Reward"].append(metrics['agg_rewards_mean'])
        agg_metrics["Median_Reward"].append(np.median(metrics['episode_reward_history']))
        agg_metrics["Standard_Deviation"].append(metrics['agg_rewards_pooled_std_dev'])
        agg_metrics["Skewness"].append(stats.skew(metrics['episode_reward_history']))
        agg_metrics["Kurtosis"].append(stats.kurtosis(metrics['episode_reward_history']))
        agg_metrics["Quartiles(Q1,Q2,Q3)"].append(np.quantile(metrics['episode_reward_history'], [0.25, 0.5, 0.75]))
        agg_metrics["IQR"].append(np.quantile(metrics['episode_reward_history'], 0.75) - np.quantile(metrics['episode_reward_history'], 0.25))
        agg_metrics["Confidence_Interval"].append(metrics['agg_rewards_confidence_interval'])
        agg_metrics["SDOM"].append(metrics['agg_rewards_sdom'])
        agg_metrics["Coeff_of_Variation"].append(metrics['agg_rewards_cv'])

    aggregate_metrics_df = pd.DataFrame.from_dict(agg_metrics, orient='index', columns=headers)

    print("Aggregate metrics calculated successfully!")

    stat_test_results = {}
    if experiment_type == 'Quantum vs Classical' and len(select_experiment_dicts) == 2:

        exp1_id = select_experiment_dicts[0]['experiment_config']['experiment_id']
        exp2_id = select_experiment_dicts[1]['experiment_config']['experiment_id']
        rewards_1 = select_experiment_dicts[0]['metrics']['episode_reward_history']
        rewards_2 = select_experiment_dicts[1]['metrics']['episode_reward_history']

        # Perform t-tests for hypothesis testing
        t_stat, p_value = perform_t_test(rewards_1, rewards_2)

        # Calculate Cohen's d for effect size
        d_value = cohen_d(rewards_1, rewards_2)

        # Perform power analysis
        effect_size = np.abs(d_value)
        alpha = 0.05  # significance level
        power = 0.8  # desired power
        ratio = len(rewards_1) / len(rewards_2)
        sample_size = TTestIndPower().solve_power(effect_size=effect_size, alpha=alpha, power=power,
                                                  ratio=ratio)

        stat_test_results["T-Statisic"] = [np.round(t_stat, 4)]
        stat_test_results["P-Value"] = [np.round(p_value, 4)]
        stat_test_results["Cohen_d"] = [np.round(d_value, 4)]
        stat_test_results["Required_Sample_Size"] = [np.round(sample_size)]

        advanced_stats_df = pd.DataFrame.from_dict(stat_test_results, orient='columns')
        advanced_stats_df.index = [f"{experiment_type}-{rl_agent}"]

        print("Advanced stats calculated successfully!")

    elif experiment_type == "Quantum N Layer" and len(select_experiment_dicts) > 2:
        # Perform one-way ANOVA
        rewards_list = [exp_dict['metrics']['episode_reward_history'] for exp_dict in select_experiment_dicts]
        levene_statistic, levene_p_value = levene(*rewards_list)
        print("Levene's Test --- Statistic:", np.round(levene_statistic,4), ", p-value:", np.round(levene_p_value,4))
        if levene_p_value < 0.05:
            print("Levene's Test --- Reject null hypothesis: Variances are not equal. Use Welch's ANOVA")
            f_stat, p_value = welchs_anova(*rewards_list)
        else:
            print("Levene's Test --- Accept null hypothesis: Variances are equal. Use regular ANOVA")
            f_stat, p_value = f_oneway(*rewards_list)

        # Calculate effect size
        # Number of groups
        k = len(rewards_list)
        # Total number of observations
        N = sum(len(rewards) for rewards in rewards_list)
        # Calculate the mean of all observations
        n_per_group = [len(rewards) for rewards in rewards_list]
        # Calculate the mean of each group
        mean_per_group = [np.mean(rewards) for rewards in rewards_list]
        # # Calculate the grand mean
        grand_mean = np.mean([reward for rewards in rewards_list for reward in rewards])
        # Calculate between-group sum of squares
        ss_between = sum(len(group) * ((mean - grand_mean) ** 2) for group, mean in zip(rewards_list, mean_per_group))
        # Calculate within-group sum of squares
        ss_within = sum(sum((obs - mean) ** 2 for obs in group) for group, mean in zip(rewards_list, mean_per_group))
        # Calculate eta squared
        eta_squared = ss_between / (ss_between + ss_within)
        # Calculate omega squared
        omega_squared = (ss_between - (k - 1) * (ss_within / (N - k))) / (
                    ss_between + ss_within + (ss_within / (N - k)))

        # power analysis for more than two groups ANOVA  (use omega_squared)

        # Initialize parameters
        effect_size = omega_squared  # Effect size
        alpha = 0.05  # Significance level
        power = 0.8  # Desired power level
        k = len(select_experiment_dicts)  # Number of groups

        # Perform power analysis
        anova_power = FTestAnovaPower()
        sample_size = anova_power.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=k)

        stat_test_results["F-Statistic"] = [np.round(f_stat, 4)]
        stat_test_results["P-Value (ANOVA)"] = [np.round(p_value, 4)]
        stat_test_results["Eta_Squared"] = [np.round(eta_squared, 4)]
        stat_test_results["Omega_Squared"] = [np.round(omega_squared, 4)]
        stat_test_results["Required_Sample_Size"] = [np.round(sample_size)]

        advanced_stats_df = pd.DataFrame.from_dict(stat_test_results, orient='columns')
        advanced_stats_df.index = [f"{experiment_type}-{rl_agent}"]

        print("Advanced stats calculated successfully!")

    else:
        print("Invalid experiment type or number of experiments!")


    # Save results to CSV if required
    if save:
        file_name_suffix = f"{experiment_type}_{rl_agent}"
        aggregate_metrics_df.to_csv(f'output_files/{file_name_suffix}_aggregate_metrics.csv', index=True)
        advanced_stats_df.to_csv(f'output_files/{file_name_suffix}_advanced_stats.csv', index=True)

    return aggregate_metrics_df, advanced_stats_df


def flatten_dict(d, parent_key='', sep='_', special_keys=None, ignore_keys=None):
    """
    Flatten a nested dictionary structure.
    Parameters:
    - d (dict): The dictionary to flatten.
    - parent_key (str): The parent key for nested dictionaries.
    - sep (str): The separator between nested keys.
    - special_keys (list): List of substrings to identify special keys that should be kept as JSON strings.
    - ignore_keys (list): List of keys to ignore during flattening.
    Returns:
    - dict: A flattened dictionary.
    """

    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        # Skip the key-value pair if the key or its parent key is in ignore_keys
        if ignore_keys and any(new_key.startswith(ik) for ik in ignore_keys):
            continue

        if isinstance(v, dict):
            if special_keys and any(sub in k for sub in special_keys):
                items[new_key] = json.dumps(v)
            else:
                items.update(flatten_dict(v, new_key, sep=sep, special_keys=special_keys, ignore_keys=ignore_keys))
        else:
            items[new_key] = v

    return items


def rearrange_fieldnames(all_fieldnames, main_keys):
    """
    Rearrange fieldnames based on the order of main keys (first-level keys in the JSON).
    Parameters:
    - all_fieldnames (set): A set of all flattened fieldnames.
    - main_keys (list): A list of main keys (first-level keys) in the original JSON.
    Returns:
    - list: A list of rearranged fieldnames.
    """

    # Initialize an empty list to hold the rearranged fieldnames
    rearranged_fieldnames = []

    for main_key in main_keys:
        # Filter all_fieldnames based on the current main_key
        filtered_fieldnames = [field for field in all_fieldnames if field.startswith(main_key)]
        # Sort and extend the rearranged_fieldnames list
        rearranged_fieldnames.extend(sorted(filtered_fieldnames))

    return rearranged_fieldnames


def combine_jsons_and_write_to_csv(json_files, output_csv_file, special_fields=None, sort_keys=None, ignore_keys=None):
    """
    Write multiple JSON files to a single CSV file.
    Parameters:
    - json_files (list): List of paths to JSON files.
    - output_csv_file (str): Path to the output CSV file.
    - special_fields (list): List of substrings to identify fields that should be kept as JSON strings in a single column.

    """

    # Initialize an empty list to hold the flattened dictionaries
    flattened_json_list = []
    # Initialize a set to hold all field names
    all_fieldnames = set()

    # Loop over each JSON file
    for json_file in json_files:
        with open(json_file, 'r') as f:
            # Load JSON data
            json_data = json.load(f)

        # Flatten the JSON data
        flattened_json_data = flatten_dict(json_data, special_keys=special_fields, ignore_keys=ignore_keys)

        # Update the set of all fieldnames
        all_fieldnames.update(flattened_json_data.keys())

        # Append the flattened JSON data to the list
        flattened_json_list.append(flattened_json_data)

        print(flattened_json_data)

    # Rearrange fieldnames if main_keys are provided
    if sort_keys:
        all_fieldnames = rearrange_fieldnames(all_fieldnames, sort_keys)

    # Write the flattened JSON list to a CSV file
    with open(output_csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        # Write the header
        writer.writeheader()
        # Write the rows
        for flattened_json_data in flattened_json_list:
            writer.writerow(flattened_json_data)

    print("Saved all experiments metrics into a single csv file")


def create_hyperparameters_csv(filepaths, save=True):
    quantum_data = []
    classical_data = []

    # Iterate through each file path to read JSON data
    for filepath in filepaths:
        print(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)

        config = data['experiment_config']
        model_details = data['model_details']
        other_params = data.get('other_parameters', {})

        # Common fields
        common_fields = [
            config['experiment_id'], config['env_name'], config['seed'],
            config['rl_agent'], config['rl_variant'], config['n_episodes'],
            config['batch_size'], config['n_actions'], config['gamma'],
            config['learning_rate'], model_details['trainable_params'],
            other_params.get('max_memory_length', None), other_params.get('epsilon', None),
            other_params.get('epsilon_min', None), other_params.get('decay_epsilon', None),
            other_params.get('steps_per_update', None),
            other_params.get('steps_per_target_update', None)
        ]

        # Append to the corresponding list based on 'rl_variant'
        if config['rl_variant'] == 'quantum':
            quantum_data.append(
                common_fields[:7] + [
                    config['n_qubits'], config['qpc_architecture'], config['n_layers']
                ] + common_fields[7:]
            )
        else:
            classical_data.append(
                common_fields[:7] + [
                    config['n_inputs'], config['n_hidden']
                ] + common_fields[7:]
            )

    # Create DataFrames and save them as CSV files
    quantum_columns = [
        "experiment_id", "env_name", "seed", "rl_agent", "rl_variant",
        "n_episodes", "batch_size", "n_qubits", "qpc_architecture",
        "n_layers", "n_actions", "gamma", "learning_rate",
        "trainable_params", "max_memory_length", "epsilon", "epsilon_min",
        "decay_epsilon", "steps_per_update", "steps_per_target_update"
    ]

    classical_columns = [
        "experiment_id", "env_name", "seed", "rl_agent", "rl_variant",
        "n_episodes", "batch_size", "n_inputs", "n_hidden", "n_actions",
        "gamma", "learning_rate", "trainable_params", "max_memory_length",
        "epsilon", "epsilon_min", "decay_epsilon", "steps_per_update",
        "steps_per_target_update"
    ]

    if save:
        pd.DataFrame(quantum_data, columns=quantum_columns).to_csv('output_files/quantum_RL_experiments_hyperparameters.csv',
                                                                   index=False)
        pd.DataFrame(classical_data, columns=classical_columns).to_csv('output_files/classical_RL_experiments_hyperparameters.csv',
                                                                       index=False)

    print("Saved hyperparameters to CSV files")


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


def filter_files_with_subdir_levels(base_dir, subdirs_with_levels, file_substring, extension):
    """
    Filters files based on the given parameters and returns a list of file paths.
    Levels are considered in the way:
    - -1 means the immediate parent directory containing the file
    - -2 means the parent directory of the directory that contains the file
    ... and so on

    This version also filters subdirectories at the same level using substring checks.

    Parameters:
    - base_dir (str): The base directory to start searching from.
    - subdirs_with_levels (list): A list containing dictionaries with subdirectories and their levels
                                  For example, [{'subdir1': -1}, {'subdir2': -2}].
    - file_substring (str): The substring to filter files by.
    - extension (str): The extension of the files to look for.

    Returns:
    - List of file paths that match the filter criteria.
    """
    # Input validation

    if not isinstance(base_dir, str) or not os.path.exists(base_dir):
        raise ValueError("Invalid base directory.")
    if not isinstance(subdirs_with_levels, list):
        raise ValueError("subdirs_with_levels should be a list.")
    if not isinstance(file_substring, str):
        raise ValueError("file_substring should be a string.")
    if not isinstance(extension, str):
        raise ValueError("extension should be a string.")

    filtered_files = []

    # Walk through the directory tree starting from base_dir
    for dirpath, _, filenames in os.walk(base_dir):
        # Split the dirpath into its components
        path_splits = dirpath.split(os.sep)

        # Initialize a flag to check if all conditions are met
        match_found = True

        # Loop through each dict in subdirs_with_levels
        for subdir_level_dict in subdirs_with_levels:

            # Type-checking: Ensure it's a dictionary
            if not isinstance(subdir_level_dict, dict):
                raise ValueError("Each element in subdirs_with_levels should be a dictionary.")

            match_dict = True  # Initialize flag for this dict
            # Check for each subdir and its level in the dict
            for subdir, level in subdir_level_dict.items():
                try:
                    # Check if the substring exists in the subdirectory at the given level
                    if subdir in path_splits[level]:
                        continue
                    else:
                        match_dict = False
                        break
                except IndexError:
                    match_dict = False
                    break

            # If all conditions in this dict are met, set match_found to True
            if match_dict:
                match_found = True
                break
            else:
                match_found = False

        if match_found:
            # Filter filenames based on substring and extension
            for filename in filenames:
                if file_substring in filename and filename.endswith(extension):
                    filtered_files.append(os.path.join(dirpath, filename))

    return filtered_files

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
    combined_dict = {k: (dict(v) if v is not None else None) for k, v in combined_dict.items()}

    temp_dict = {}
    # Average numerical metrics and keep unique config values
    all_runs_np = None
    for main_key, sub_dict in combined_dict.items():
        if main_key in ['metrics', 'timings']:

            if isinstance(sub_dict, dict):
                for sub_key, value_list in sub_dict.items():
                    if all(isinstance(v, (int, float)) for v in value_list):
                        if sub_key in ['episode_count', 'episode_rewards_min', 'episode_rewards_max', 'episode_rewards_mean', 'episode_rewards_median', 'episode_rewards_std',
                                      'episode_rewards_q1', 'episode_rewards_q3', 'episode_rewards_iqr']:
                            means = combined_dict[main_key]['episode_rewards_mean']
                            std_devs = combined_dict[main_key]['episode_rewards_std']
                            n_per_run = combined_dict[main_key]['episode_count']

                            N = sum(combined_dict[main_key]['episode_count'])

                            # Calculate overall mean
                            overall_mean = np.average(means, weights=n_per_run)

                            # Calculate pooled standard deviation
                            numerator = sum((n - 1) * (std_dev ** 2) for n, std_dev in zip(n_per_run, std_devs))
                            pooled_std_dev = math.sqrt(numerator / (N - len(n_per_run)))

                            # Calculate Q1 and Q3
                            agg_q1 = np.mean(combined_dict[main_key]['episode_rewards_q1'])
                            agg_q3 = np.mean(combined_dict[main_key]['episode_rewards_q3'])

                            # Calculate IQR
                            agg_iqr = agg_q3 - agg_q1

                            # Calculate SDOM
                            sdom = pooled_std_dev / math.sqrt(len(n_per_run))

                            # Calculate Coefficient of Variation (CV)
                            cv = (pooled_std_dev / overall_mean) * 100

                            # Calculate Confidence Interval
                            confidence_level = 0.95
                            z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                            margin_error = z_value * (pooled_std_dev / math.sqrt(N))
                            confidence_interval = (np.round(overall_mean - margin_error,2), np.round(overall_mean + margin_error,2))

                            #store agg metrics in temp dict
                            temp_dict[main_key] = {}
                            temp_dict[main_key]['agg_rewards_mean'] = np.round(overall_mean,2)
                            temp_dict[main_key]['agg_rewards_pooled_std_dev'] = np.round(pooled_std_dev,2)
                            temp_dict[main_key]['agg_rewards_q1'] = int(np.round(agg_q1))
                            temp_dict[main_key]['agg_rewards_q3'] = int(np.round(agg_q3))
                            temp_dict[main_key]['agg_rewards_iqr'] = int(np.round(agg_iqr))
                            temp_dict[main_key]['agg_rewards_sdom'] = np.round(sdom,2)
                            temp_dict[main_key]['agg_rewards_cv'] = np.round(cv,2)
                            temp_dict[main_key]['agg_rewards_confidence_interval'] = confidence_interval

                        else:
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

    for main_key, sub_dict in temp_dict.items():
        for sub_key, value in sub_dict.items():
            combined_dict[main_key][sub_key] = value

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

def smooth_curve(y, exp_id, smoothing_method="EMA", smoothing_level=0.9, save=False, save_file_name="smoothed_data_{}.json".format(datetime.now().strftime("%Y%m%d-%H%M%S"))):
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

        # Plot individual reward history with alpha=0.4 and light blue colour
        plt.plot(episode_rewards, alpha=0.3, linewidth=1, color='lightblue')

    # Single legend entry for all reruns
    plt.plot([], [], alpha=0.4, color='lightblue', linewidth=1, label='All reruns({})'.format(n_exps))

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


def plot_multiple_experiments(multi_exp_reward_histories, exp_ids, experiment_names, smoothing_method="EMA", smoothing_level=0.9,
                              save=True, fig_size=(10, 7), plot_title="Rewards over Episodes"):
    """
    Plots multiple experiment reward histories in one plot.
    """
    basic_text_size = 12
    line_width = 2
    text_sizes = adjust_text_size(basic_text_size)  # Adjust text sizes
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Different colors for different curves  - [blue, green, red, cyan, magenta, yellow, black]
    plt.figure(figsize=fig_size)

    for idx, (exp_rewards, exp_id) in enumerate(zip(multi_exp_reward_histories, exp_ids)):
        color = colors[idx % len(colors)]
        print("color: ", color)

        # Smooth the average rewards
        smoothed_rewards = smooth_curve(exp_rewards, exp_id, smoothing_method=smoothing_method,
                                        smoothing_level=smoothing_level, save=False)

        # Plot original curve with light shade
        plt.plot(exp_rewards, color=color, alpha=0.1, linewidth=1)    # label=f"{exp_id} Original"
        if len(exp_ids) == 2 and "Quantum" in experiment_names[exp_id]:
            legend_str = " ".join(experiment_names[exp_id].split(" ")[:-2])
        else:
            legend_str = experiment_names[exp_id]

        #---check how many run directories under each exp_id directory to get number of runs for each experiment
        n_runs = len(glob.glob(os.path.join(exp_id, "run*")))

        legend_str += " ({} runs)".format(n_runs)
        # Plot smoothed curve with dark shade
        plt.plot(smoothed_rewards, label=f"{legend_str}", color=color, linewidth=2)

    plt.xlabel('Episodes', fontsize=text_sizes["label_size"])
    plt.ylabel('Averaged Rewards', fontsize=text_sizes["label_size"])
    plt.title(plot_title, fontsize=text_sizes["title_size"])
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=text_sizes["legend_size"])
    plt.legend(loc='lower right', fontsize=text_sizes["legend_size"])

    # Adjust tick sizes
    plt.xticks(fontsize=text_sizes["tick_size"])
    plt.yticks(fontsize=text_sizes["tick_size"])

    plot_save_file_name = "output_files/{}.png".format(plot_title)  # datetime.now().strftime("%Y%m%d-%H%M%S")
    if save:
        plt.savefig(plot_save_file_name) #, bbox_inches='tight', pad_inches=0.1)
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


