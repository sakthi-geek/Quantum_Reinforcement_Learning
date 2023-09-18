import os
import helper
from datetime import datetime
import json
import pandas as pd
from pathlib import Path



def main():
    # # Load experiment configurations
    # with open('Classical_Quantum_RL_experiments.json', 'r') as file:
    #     experiments_config = json.load(file)
    #
    # all_experiment_ids = ["experiment_1", "experiment_2a", "experiment_2b", "experiment_2c",
    #                                             "experiment_3", "experiment_4a", "experiment_4b", "experiment_4c",
    #                                             "experiment_5", "experiment_6a", "experiment_6b", "experiment_6c"]
    #
    # # exp_id = "experiment_6c"
    # # exp_config = None
    #
    # for exp_id in all_experiment_ids:
    #     exp_config = None
    #     # Iterate through the nested dictionaries to find the target experiment config
    #     for exp in experiments_config.get('experiments', []):
    #         baseline_rl_experiments = exp.get('baseline_rl_experiments', {})
    #         if exp_id in baseline_rl_experiments:
    #             exp_config = baseline_rl_experiments[exp_id]
    #             break
    #
    #     # Aggregate results from all reruns of an experiment
    #     print("exp_id: ", exp_id)
    #     base_dir = os.getcwd()
    #     print("base_dir: ", base_dir)
    #     subdirs_with_levels = [{'run_': -1, exp_id: -2}]
    #     print("subdirs_with_levels: ", subdirs_with_levels)
    #     substring = 'outcome'
    #     print("file substring: ", substring)
    #     extension = '.json'
    #     print("file extension: ", extension)
    #
    #     outcome_files = helper.filter_files_with_subdir_levels(base_dir, subdirs_with_levels, substring, extension)
    #     print(len(outcome_files))
    #     print("filtered_outcome_files: ", outcome_files)
    #
    #     experiment_outcomes_list = []
    #     exp_rewards_history_list = []
    #     for outcome_file in outcome_files:
    #         with open(outcome_file, 'r') as file:
    #             exp_outcome = json.load(file)
    #             exp_rewards_history = exp_outcome["metrics"]["episode_reward_history"]
    #             exp_rewards_history_list.append(exp_rewards_history)
    #             experiment_outcomes_list.append(exp_outcome)
    #
    #     helper.plot_averaged_rewards(exp_rewards_history_list, exp_id, save=False,
    #                                  plot_title=f"{exp_config['rl_variant']} {exp_config['rl_agent']} averaged rewards")
    #
    #     averaged_reward_history = helper.compute_averaged_rewards(exp_rewards_history_list, method='plateau',
    #                                                              fill_value=None)
    #     helper.plot_smoothed_rewards(averaged_reward_history, exp_id, smoothing_method="EMA",
    #                                  smoothing_level=0.9, save=False, prefix_str="averaged",
    #                                  plot_title=f"{exp_config['rl_variant']} {exp_config['rl_agent']} averaged rewards 2")
    #
    #     combined_outcome_dict = helper.combine_experiment_outcomes(experiment_outcomes_list, exp_id, combining_strategy = "mean")
    #     helper.plot_smoothed_rewards(combined_outcome_dict["metrics"]["episode_reward_history"], exp_id, smoothing_method="EMA",
    #                                      smoothing_level=0.9, save=False,
    #                                     plot_title = f"{exp_config['rl_variant']} {exp_config['rl_agent']} averaged rewards 3")

    #--------------------------------------------------------------------------------------------------
    # print("=====================================")
    # # Plot multiple experiments
    # rl_agent = "Actor_Critic"     # Policy_Gradient_REINFORCE, Deep_Q_Learning, Actor_Critic
    # print("rl_agent: ", rl_agent)
    #
    # experiment_type = "Quantum N Layer"     # Quantum vs Classical, Quantum N Layer
    # print("experiment_type: ", experiment_type)
    #
    # experiment_names = {}
    # all_experiment_ids = ["experiment_1", "experiment_2a", "experiment_2b", "experiment_2c",
    #                       "experiment_3", "experiment_4a", "experiment_4b", "experiment_4c",
    #                       "experiment_5", "experiment_6a", "experiment_6b", "experiment_6c"]
    # experiment_names["experiment_1"] = "Policy Gradient REINFORCE - Classical"
    # experiment_names["experiment_2a"] = "Policy Gradient REINFORCE - Quantum 5 Layers"
    # experiment_names["experiment_2b"] = "Policy Gradient REINFORCE - Quantum 4 Layers"
    # experiment_names["experiment_2c"] = "Policy Gradient REINFORCE - Quantum 3 Layers"
    # experiment_names["experiment_3"] = "Deep Q Learning - Classical"
    # experiment_names["experiment_4a"] = "Deep Q Learning - Quantum 5 Layers"
    # experiment_names["experiment_4b"] = "Deep Q Learning - Quantum 4 Layers"
    # experiment_names["experiment_4c"] = "Deep Q Learning - Quantum 3 Layers"
    # experiment_names["experiment_5"] = "Actor Critic - Classical"
    # experiment_names["experiment_6a"] = "Actor Critic - Quantum 5 Layers"
    # experiment_names["experiment_6b"] = "Actor Critic - Quantum 4 Layers"
    # experiment_names["experiment_6c"] = "Actor Critic - Quantum 3 Layers"
    #
    # rl_agent_list = ["Policy_Gradient_REINFORCE", "Deep_Q_Learning", "Actor_Critic"]
    # selection_type_list = ["Quantum vs Classical", "Quantum N Layer"]
    #
    # # Plot all combinations of rl_agent and experiment_type
    # for experiment_type in selection_type_list:
    #     for rl_agent in rl_agent_list:
    #         select_experiment_ids = []
    #         if experiment_type == "Quantum vs Classical":
    #             if rl_agent == "Policy_Gradient_REINFORCE":
    #                 select_experiment_ids = ["experiment_1", "experiment_2a"]
    #             elif rl_agent == "Deep_Q_Learning":
    #                 select_experiment_ids = ["experiment_3", "experiment_4a"]
    #             elif rl_agent == "Actor_Critic":
    #                 select_experiment_ids = ["experiment_5", "experiment_6a"]
    #         elif experiment_type == "Quantum N Layer":
    #             if rl_agent == "Policy_Gradient_REINFORCE":
    #                 select_experiment_ids = ["experiment_2a", "experiment_2b", "experiment_2c"]
    #             elif rl_agent == "Deep_Q_Learning":
    #                 select_experiment_ids = ["experiment_4a", "experiment_4b", "experiment_4c"]
    #             elif rl_agent == "Actor_Critic":
    #                 select_experiment_ids = ["experiment_6a", "experiment_6b", "experiment_6c"]
    #
    #         print("select_experiment_ids: ", select_experiment_ids)
    #
    #
    #         print("--------------------")
    #
    #         # Aggregate results from all reruns of an experiment
    #         base_dir = os.getcwd()
    #         print("base_dir: ", base_dir)
    #         subdirs_with_levels_list = []
    #         for exp_id in select_experiment_ids:
    #             subdirs_with_levels = {}
    #             subdirs_with_levels[exp_id] = -1
    #             subdirs_with_levels_list.append(subdirs_with_levels)
    #         print("subdirs_with_levels list: ", subdirs_with_levels_list)
    #         substring = 'combined'
    #         print("file substring: ", substring)
    #         extension = '.json'
    #         print("file extension: ", extension)
    #
    #         select_exp_outcome_files = helper.filter_files_with_subdir_levels(base_dir, subdirs_with_levels_list, substring, extension)
    #         print(len(select_exp_outcome_files))
    #         print("filtered_outcome_files: ", select_exp_outcome_files)
    #
    #         multi_exp_reward_histories = []
    #         for select_exp_outcome_file in select_exp_outcome_files:
    #             with open(select_exp_outcome_file, 'r') as file:
    #                 select_exp_outcome = json.load(file)
    #                 exp_rewards_history = select_exp_outcome["metrics"]["episode_reward_history"]
    #                 multi_exp_reward_histories.append(exp_rewards_history)
    #
    #         print(len(multi_exp_reward_histories))
    #         print("multi_exp_reward_histories: ", multi_exp_reward_histories)
    #
    #
    #         helper.plot_multiple_experiments(multi_exp_reward_histories, select_experiment_ids, experiment_names, save=True,
    #                             plot_title=f"{rl_agent}-{experiment_type} - averaged rewards")

    #-------------------------------------------------------------------------------------
    # # filter all outcome files from all experiments and reruns
    # base_dir = os.getcwd()
    # print("base_dir: ", base_dir)
    # subdirs_with_levels_list = []
    # for exp_id in ["experiment_4b", "experiment_4c"]:
    #     subdirs_with_levels = {}
    #     subdirs_with_levels[exp_id] = -1
    #     subdirs_with_levels_list.append(subdirs_with_levels)
    #
    #     subdirs_with_levels_2 = {}
    #     subdirs_with_levels_2["run"] = -1
    #     subdirs_with_levels_2[exp_id] = -2
    #     subdirs_with_levels_list.append(subdirs_with_levels_2)
    #
    # print("subdirs_with_levels list: ", subdirs_with_levels_list)
    # substring = 'outcome'
    # print("file substring: ", substring)
    # extension = '.json'
    # print("file extension: ", extension)
    #
    # select_outcome_files = helper.filter_files_with_subdir_levels(base_dir, subdirs_with_levels_list, substring,
    #                                                                   extension)
    # print(len(select_outcome_files))
    # print("filtered_outcome_files: ", select_outcome_files)

    # # update json files
    # helper.update_json_files(select_outcome_files, save=True)

    #-------------------------------------------------------------------------------------
    # # # filter all combined outcome files from all experiments
    # experiment_names = {}
    # all_experiment_ids = ["experiment_1", "experiment_2a", "experiment_2b", "experiment_2c",
    #                       "experiment_3", "experiment_4a", "experiment_4b", "experiment_4c",
    #                       "experiment_5", "experiment_6a", "experiment_6b", "experiment_6c"]
    # experiment_names["experiment_1"] = "Policy Gradient REINFORCE - Classical"
    # experiment_names["experiment_2a"] = "Policy Gradient REINFORCE - Quantum 5 Layers"
    # experiment_names["experiment_2b"] = "Policy Gradient REINFORCE - Quantum 4 Layers"
    # experiment_names["experiment_2c"] = "Policy Gradient REINFORCE - Quantum 3 Layers"
    # experiment_names["experiment_3"] = "Deep Q Learning - Classical"
    # experiment_names["experiment_4a"] = "Deep Q Learning - Quantum 5 Layers"
    # experiment_names["experiment_4b"] = "Deep Q Learning - Quantum 4 Layers"
    # experiment_names["experiment_4c"] = "Deep Q Learning - Quantum 3 Layers"
    # experiment_names["experiment_5"] = "Actor Critic - Classical"
    # experiment_names["experiment_6a"] = "Actor Critic - Quantum 5 Layers"
    # experiment_names["experiment_6b"] = "Actor Critic - Quantum 4 Layers"
    # experiment_names["experiment_6c"] = "Actor Critic - Quantum 3 Layers"
    #-------------------------------------------------------------------------------------

    # base_dir = os.getcwd()
    # print("base_dir: ", base_dir)
    # subdirs_with_levels_list = []
    # for exp_id in all_experiment_ids:
    #     subdirs_with_levels = {}
    #     subdirs_with_levels[exp_id] = -1
    #     subdirs_with_levels_list.append(subdirs_with_levels)
    #
    # print("subdirs_with_levels list: ", subdirs_with_levels_list)
    # substring = 'combined'
    # print("file substring: ", substring)
    # extension = '.json'
    # print("file extension: ", extension)
    #
    # select_combined_outcome_files = helper.filter_files_with_subdir_levels(base_dir, subdirs_with_levels_list, substring,
    #                                                                     extension)
    # print(len(select_combined_outcome_files))
    # print("filtered_outcome_files: ", select_combined_outcome_files)

    # helper.create_hyperparameters_csv(select_combined_outcome_files, save=False) # DO NOT SAVE

    #-------------------------------------------------------------------------------------
    # # Initialize empty list to store metrics
    # all_experiment_metrics = []
    #
    # # Assuming `all_experiments` is your list of dictionaries for each experiment
    # all_experiments = [...]  # Replace with your actual data
    #
    # # Loop through each experiment to calculate metrics
    # for experiment in all_experiments:
    #     exp_id = experiment['experiment_config']['experiment_id']
    #     rewards = np.array(experiment['metrics']['episode_reward_history'])
    #
    #     # Calculate metrics
    #     mean_reward = np.mean(rewards)
    #     std_dev_reward = np.std(rewards)
    #     lower_bound, upper_bound = confidence_interval(rewards)
    #     sdom = standard_deviation_of_mean(std_dev_reward, len(rewards))
    #     cv = coefficient_of_variation(mean_reward, std_dev_reward)
    #
    #     # Store in a dictionary
    #     experiment_metrics = {
    #         'Experiment_ID': exp_id,
    #         'Mean_Reward': mean_reward,
    #         'Standard_Deviation': std_dev_reward,
    #         'Lower_Bound_CI': lower_bound,
    #         'Upper_Bound_CI': upper_bound,
    #         'SDOM': sdom,
    #         'CV': cv
    #     }
    #
    #     # Append to list
    #     all_experiment_metrics.append(experiment_metrics)
    #
    # # Convert to DataFrame for easy CSV export
    # df = pd.DataFrame(all_experiment_metrics)
    #
    # # Write to CSV
    # df.to_csv('RL_experiment_metrics.csv', index=False)

    #--------------------------------------------------------------------------------------------------
    # # filter all combined outcome files from all experiments
    # base_dir = os.getcwd()
    # print("base_dir: ", base_dir)
    # subdirs_with_levels = [{'experiment':-1 , 'Quantum_RL':-2}]
    # substring = 'combined'
    # extension = '.json'
    #
    # filtered_files = helper.filter_files_with_subdir_levels(base_dir, subdirs_with_levels, substring, extension)
    # print(len(filtered_files))
    # print("filtered_files: ", filtered_files)
    #
    # output_csv_file = "combined_metrics_{}.csv".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    # special_fields = ["learning_rate"]
    # sort_keys = ['experiment', 'model_details', 'timings', 'metrics']
    # ignore_keys = ["other_parameters"]
    #
    # helper.combine_jsons_and_write_to_csv(filtered_files, output_csv_file, special_fields=special_fields, sort_keys=sort_keys, ignore_keys=ignore_keys)

    #--------------------------------------------------------------------------------------------------
    # # # filter all combined outcome files from all experiments
    # experiment_names = {}
    # all_experiment_ids = ["experiment_1", "experiment_2a", "experiment_2b", "experiment_2c",
    #                       "experiment_3", "experiment_4a", "experiment_4b", "experiment_4c",
    #                       "experiment_5", "experiment_6a", "experiment_6b", "experiment_6c"]
    # experiment_names["experiment_1"] = "Policy Gradient REINFORCE - Classical"
    # experiment_names["experiment_2a"] = "Policy Gradient REINFORCE - Quantum 5 Layers"
    # experiment_names["experiment_2b"] = "Policy Gradient REINFORCE - Quantum 4 Layers"
    # experiment_names["experiment_2c"] = "Policy Gradient REINFORCE - Quantum 3 Layers"
    # experiment_names["experiment_3"] = "Deep Q Learning - Classical"
    # experiment_names["experiment_4a"] = "Deep Q Learning - Quantum 5 Layers"
    # experiment_names["experiment_4b"] = "Deep Q Learning - Quantum 4 Layers"
    # experiment_names["experiment_4c"] = "Deep Q Learning - Quantum 3 Layers"
    # experiment_names["experiment_5"] = "Actor Critic - Classical"
    # experiment_names["experiment_6a"] = "Actor Critic - Quantum 5 Layers"
    # experiment_names["experiment_6b"] = "Actor Critic - Quantum 4 Layers"
    # experiment_names["experiment_6c"] = "Actor Critic - Quantum 3 Layers"
    #
    # # rl_agent = "Policy_Gradient_REINFORCE"  # Policy_Gradient_REINFORCE, Deep_Q_Learning, Actor_Critic
    # # print("rl_agent: ", rl_agent)
    # #
    # # experiment_type = "Quantum vs Classical"  # Quantum vs Classical, Quantum N Layer
    # # print("experiment_type: ", experiment_type)
    #
    # rl_agent_list = ["Policy_Gradient_REINFORCE", "Deep_Q_Learning", "Actor_Critic"]
    # experiment_type_list = ["Quantum vs Classical", "Quantum N Layer"]
    #
    # # Plot all combinations of rl_agent and experiment_type
    # for experiment_type in experiment_type_list:
    #     for rl_agent in rl_agent_list:
    #         select_experiment_ids = []
    #         if experiment_type == "Quantum vs Classical":
    #             if rl_agent == "Policy_Gradient_REINFORCE":
    #                 select_experiment_ids = ["experiment_1", "experiment_2a"]
    #             elif rl_agent == "Deep_Q_Learning":
    #                 select_experiment_ids = ["experiment_3", "experiment_4a"]
    #             elif rl_agent == "Actor_Critic":
    #                 select_experiment_ids = ["experiment_5", "experiment_6a"]
    #         elif experiment_type == "Quantum N Layer":
    #             if rl_agent == "Policy_Gradient_REINFORCE":
    #                 select_experiment_ids = ["experiment_2a", "experiment_2b", "experiment_2c"]
    #             elif rl_agent == "Deep_Q_Learning":
    #                 select_experiment_ids = ["experiment_4a", "experiment_4b", "experiment_4c"]
    #             elif rl_agent == "Actor_Critic":
    #                 select_experiment_ids = ["experiment_6a", "experiment_6b", "experiment_6c"]

    #         print("select_experiment_ids: ", select_experiment_ids)
    #
    #         # filter all combined outcome files from selected experiments
    #         base_dir = os.getcwd()
    #         print("base_dir: ", base_dir)
    #         subdirs_with_levels_list = []
    #         for exp_id in select_experiment_ids:
    #             subdirs_with_levels = {}
    #             subdirs_with_levels[exp_id] = -1
    #             subdirs_with_levels_list.append(subdirs_with_levels)
    #
    #         print("subdirs_with_levels list: ", subdirs_with_levels_list)
    #         substring = 'combined'
    #         print("file substring: ", substring)
    #         extension = '.json'
    #         print("file extension: ", extension)
    #
    #         select_combined_outcome_files = helper.filter_files_with_subdir_levels(base_dir, subdirs_with_levels_list, substring,
    #                                                                                 extension)
    #         print(len(select_combined_outcome_files))
    #         print("filtered_outcome_files: ", select_combined_outcome_files)
    #
    #         #create a list of dicts from the json files
    #         select_combined_outcome_dict_list = []
    #         for select_combined_outcome_file in select_combined_outcome_files:
    #             with open(select_combined_outcome_file, 'r') as file:
    #                 select_combined_outcome_dict = json.load(file)
    #                 select_combined_outcome_dict_list.append(select_combined_outcome_dict)
    #
    #         print("Experiment_Type: ", experiment_type)
    #         print("RL_Agent: ", rl_agent)
    #         print(len(select_combined_outcome_dict_list))
    #
    #         helper.calculate_agg_metrics_and_advanced_stats(select_combined_outcome_dict_list, experiment_type, rl_agent, experiment_names, save=True)
    #--------------------------------------------------------------------------------------------------
    #
    # base_dir = os.getcwd()
    # print("base_dir: ", base_dir)
    # subdirs_with_levels_list = []
    #
    # # subdirs_with_levels = {}
    # # subdirs_with_levels["RL_Experiment_Analysis"] = -1
    # # subdirs_with_levels["Quantum_RL_Experiments"] = -2
    # # subdirs_with_levels_list.append(subdirs_with_levels)
    #
    # subdirs_with_levels = {}
    # subdirs_with_levels["RL_Experiment_Analysis"] = -1
    # subdirs_with_levels["Quantum_RL_Experiments"] = -2
    # subdirs_with_levels_list.append(subdirs_with_levels)
    #
    # print("subdirs_with_levels list: ", subdirs_with_levels_list)
    # substring = 'metrics'
    # print("file substring: ", substring)
    # extension = '.csv'
    # print("file extension: ", extension)
    #
    # output_csv_files = helper.filter_files_with_subdir_levels(base_dir, subdirs_with_levels_list, substring,
    #                                                                         extension)
    # print(len(output_csv_files))
    # print("filtered_outcome_files: ", output_csv_files)
    # # convert csv into latex table
    # for csv_file in output_csv_files:
    #     print("=====================================")
    #     print("csv_file: ", csv_file.split("/")[-1])
    #     df = pd.read_csv(csv_file)
    #     print(df.to_latex(index=False, ))
    #     print("=====================================")

#==================================================================================================
    # all_experiment_ids = ["experiment_1", "experiment_2a", "experiment_2b", "experiment_2c",
    #                                             "experiment_3", "experiment_4a", "experiment_4b", "experiment_4c",
    #                                             "experiment_5", "experiment_6a", "experiment_6b", "experiment_6c"]
    # exp_names_dict = {}
    # exp_names_dict["experiment_1"] = "Classical REINFORCE"
    # exp_names_dict["experiment_2a"] = "Quantum REINFORCE(5 layer)"
    # exp_names_dict["experiment_2b"] = "Quantum REINFORCE(4 layer)"
    # exp_names_dict["experiment_2c"] = "Quantum REINFORCE(3 layer)"
    # exp_names_dict["experiment_3"] = "Classical DQN"
    # exp_names_dict["experiment_4a"] = "Quantum DQN(5 layer)"
    # exp_names_dict["experiment_4b"] = "Quantum DQN(4 layer)"
    # exp_names_dict["experiment_4c"] = "Quantum DQN(3 layer)"
    # exp_names_dict["experiment_5"] = "Classical Actor Critic"
    # exp_names_dict["experiment_6a"] = "Quantum Actor Critic(5 layer)"
    # exp_names_dict["experiment_6b"] = "Quantum Actor Critic(4 layer)"
    # exp_names_dict["experiment_6c"] = "Quantum Actor Critic(3 layer)"
    #
    # all_experiment_outcomes_dict = {}
    # for exp_id in all_experiment_ids:
    #     print("======================================")
    #     print("exp_id: ", exp_id)
    #     # Aggregate results from all reruns of an experiment
    #     base_dir = os.getcwd()
    #     print("base_dir: ", base_dir)
    #     subdirs_with_levels_list = []
    #
    #     subdirs_with_levels = {}
    #     subdirs_with_levels["run_"] = -1
    #     subdirs_with_levels[exp_id] = -2
    #     subdirs_with_levels_list.append(subdirs_with_levels)
    #
    #     print("subdirs_with_levels list: ", subdirs_with_levels_list)
    #     substring = 'outcomes'
    #     print("file substring: ", substring)
    #     extension = '.json'
    #     print("file extension: ", extension)
    #
    #     exp_all_runs_outcome_files = helper.filter_files_with_subdir_levels(base_dir, subdirs_with_levels_list, substring,
    #                                                               extension)
    #     print(len(exp_all_runs_outcome_files))
    #     print("filtered_outcome_files: ", exp_all_runs_outcome_files)
    #
    #     all_experiment_outcomes_dict[exp_id] = exp_all_runs_outcome_files
    #
    # print(all_experiment_outcomes_dict)
    #
    # helper.calculate_metrics_with_descriptive_stats(all_experiment_outcomes_dict, exp_names_dict, save=True)
    # # #
    # # helper.extract_hyperparameters(all_experiment_outcomes_dict, save=False)



if __name__ == "__main__":
    main()