# from helper import save_results, save_plots, save_model
# from environments import create_environment

# Importing classical agents
from Quantum_RL_Experiments.classical_RL_agents.REINFORCE_classical import REINFORCEAgent
from Quantum_RL_Experiments.classical_RL_agents.deep_q_learning_classical import DeepQLearningClassical
from Quantum_RL_Experiments.classical_RL_agents.actor_critic_classical import ActorCriticClassical

# Importing quantum agents
from Quantum_RL_Experiments.quantum_RL_agents.policy_gradient_quantum import PolicyGradientQuantum
from Quantum_RL_Experiments.quantum_RL_agents.deep_q_learning_quantum import DeepQLearningQuantum
from Quantum_RL_Experiments.quantum_RL_agents.actor_critic_quantum import ActorCriticQuantum

import json
import time
import os
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from helper import NumpyEncoder
import helper


def run_experiments(experiments_config, reruns_per_experiment=1, start_from_rerun_id=1, skip_whole_training_block=False, disable_only_training_code=False):

    experiment_ids = [exp_id for exp_id, exp_config in experiments_config.items()]
    print('Running experiments: {}'.format(experiment_ids))
    for experiment_id in experiment_ids:
        if not os.path.exists(experiment_id):
            ## shutil.rmtree(experiment_folder)
            os.makedirs(experiment_id)

    # Iterate over all the experiments in the config file
    for exp_id, exp_config in experiments_config.items():
        # Log experiment config
        with open(os.path.join(exp_id, '{}_config.json'.format(exp_id)), 'w') as file:
            json.dump(exp_config, file, indent=4)
        print("Experiment config saved to file")

        for rerun_id in range(start_from_rerun_id, start_from_rerun_id+reruns_per_experiment):
            print("---", rerun_id)
            exp_config['rerun_id'] = rerun_id
            #cretate rerun_id folders if not present
            run_dir = os.path.join(exp_id, str("run_{}".format(rerun_id)))
            if not disable_only_training_code:
                if not os.path.exists(run_dir):
                    os.makedirs(run_dir)

            print('Running experiment: {}, rerun_id: {}'.format(exp_id, rerun_id))
            start_time = time.time()
            print('Start time: {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

            # Unpack parameters
            rl_agent = exp_config.get('rl_agent')                           # RL agent to use
            rl_variant = exp_config.get('rl_variant')                       # Classical or Quantum
            exploration_strategy = exp_config.get('exploration_strategy')   # Exploration strategy
            n_episodes = exp_config.get('n_episodes')                       # Training episodes
            batch_size = exp_config.get('batch_size')                       # Batch size
            learning_rate = exp_config.get('learning_rate')                 # Learning rate
            gamma = exp_config.get('gamma')                                 # Discount factor
            max_steps_per_episode = exp_config.get('max_steps_per_episode') # Max steps per episode
            #-----------------Classical RL specific parameters-----------------
            n_inputs = exp_config.get('n_inputs')                           # Classical specific
            n_hidden = exp_config.get('n_hidden')                           # Classical specific
            n_actions = exp_config.get('n_actions')                         # Classical specific

            #-----------------Quantum RL specific parameters-----------------
            n_qubits = exp_config.get('n_qubits')                           # Quantum specific
            n_layers = exp_config.get('n_layers')                           # Quantum specific
            qpc_architecture = exp_config.get('qpc_architecture')           # Quantum specific
            exp_config["seed"] = 39+(rerun_id*100)
            exp_config["env_name"] = "CartPole-v1"

            print("Experiment ID: ", exp_id)
            print("rerun_id: ", rerun_id)
            print("seed: ", exp_config["seed"])
            print("env_name: ", exp_config["env_name"])
            print("rl_agent: ", rl_agent)
            print("rl_variant: ", rl_variant)
            print("exploration_strategy: ", exploration_strategy)
            print("n_episodes: ", n_episodes)
            print("batch_size: ", batch_size)
            print("max_steps_per_episode: ", max_steps_per_episode)
            print("learning_rate: ", learning_rate)
            print("gamma: ", gamma)
            print("n_inputs: ", n_inputs)
            print("n_hidden: ", n_hidden)
            print("n_actions: ", n_actions)
            print("n_qubits: ", n_qubits)
            print("n_layers: ", n_layers)
            print("qpc_architecture: ", qpc_architecture)

            if not skip_whole_training_block:
                # Initialize Agent
                agent = None
                if rl_variant == 'classical':
                    if rl_agent == 'policy_gradient_reinforce':
                        agent = REINFORCEAgent(env_name=exp_config["env_name"], seed=exp_config["seed"] ,
                                               n_inputs=n_inputs, n_hidden=n_hidden, n_actions=n_actions,
                                               gamma=gamma, n_episodes=n_episodes, batch_size=batch_size,
                                               learning_rate=learning_rate)
                    elif rl_agent == 'deep_q_learning':
                        agent = DeepQLearningClassical(env_name='CartPole-v1', seed=exp_config["seed"],
                                                       n_inputs=n_inputs, n_hidden=n_hidden, n_actions=n_actions,
                                                       gamma=gamma, n_episodes=n_episodes, batch_size=batch_size,
                                                       learning_rate=learning_rate)
                    elif rl_agent == 'actor_critic':
                        agent = ActorCriticClassical(env_name=exp_config["env_name"], seed=exp_config["seed"],
                                                     n_inputs=n_inputs, n_hidden=n_hidden, n_actions=n_actions,
                                                     gamma=gamma, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode,
                                                     learning_rate=learning_rate)
                elif rl_variant == 'quantum':
                    if rl_agent == 'policy_gradient':
                        agent = PolicyGradientQuantum(env_name=exp_config["env_name"], seed=exp_config["seed"],
                                                      n_qubits=n_qubits, n_layers=n_layers, n_actions=n_actions,
                                                      gamma=gamma, n_episodes=n_episodes, batch_size=batch_size,
                                                      learning_rate=learning_rate)
                    elif rl_agent == 'deep_q_learning':
                        agent = DeepQLearningQuantum(env_name=exp_config["env_name"], seed=exp_config["seed"],
                                                     n_qubits=n_qubits, n_layers=n_layers, n_actions=n_actions,
                                                     gamma=gamma, n_episodes=n_episodes, batch_size=batch_size,
                                                     learning_rate=learning_rate)
                    elif rl_agent == 'actor_critic':
                        agent = ActorCriticQuantum(env_name=exp_config["env_name"], seed=exp_config["seed"],
                                                   n_qubits=n_qubits, n_layers=n_layers, n_actions=n_actions,
                                                   gamma=gamma, n_episodes=n_episodes, max_steps_per_episode=max_steps_per_episode,
                                                   learning_rate=learning_rate)

                print(agent)


                # Training
                training_start_time = time.time()
                print("Training start time: {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
                if not disable_only_training_code:
                    agent.train()
                training_end_time = time.time()
                print("Training end time: {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

                # # Inference
                # inference_start_time = time.time()
                # agent.play()
                # inference_end_time = time.time()

                try:
                    if hasattr(agent, 'model'):
                        model_summary_file_name = '{}_{}_model_summary.txt'.format(rl_variant, rl_agent)
                        # check if file already present and skip if present
                        # if os.path.exists(os.path.join(exp_id, model_summary_file_name)):
                        #     print("Skipped saving model summary due to file already present")
                        # else:
                        # save model summary to file
                        with open(os.path.join(exp_id, model_summary_file_name), 'w') as fh:
                            # Pass the file handle in as a lambda function to make it callable
                            agent.model.summary(print_fn=lambda x: fh.write(x + '\n'))
                        print("Saved model summary to file")
                    elif hasattr(agent, 'actor_model') and hasattr(agent, 'critic_model'):
                        actor_model_summary_file_name = '{}_{}_actor_model_summary.txt'.format(rl_variant, rl_agent)
                        with open(os.path.join(exp_id, actor_model_summary_file_name), 'w') as fh:
                            # Pass the file handle in as a lambda function to make it callable
                            agent.actor_model.summary(print_fn=lambda x: fh.write(x + '\n'))

                        critic_model_summary_file_name = '{}_{}_critic_model_summary.txt'.format(rl_variant, rl_agent)
                        with open(os.path.join(exp_id, critic_model_summary_file_name), 'w') as fh:
                            # Pass the file handle in as a lambda function to make it callable
                            agent.critic_model.summary(print_fn=lambda x: fh.write(x + '\n'))
                        print("Saved model summary to file")
                    else:
                        print("Skipped saving model summary due to model not found")

                except:
                    print("Skipped saving model summary due to error")

                try:
                    if hasattr(agent, 'model'):
                        model_arch_file_name = '{}_{}_model_architecture.png'.format(rl_variant, rl_agent)
                        # check if file already present and skip if present
                        # if os.path.exists(os.path.join(exp_id, model_arch_file_name)):
                        #     print("Skipped saving model architecture due to file already present")
                        # else:
                        # Save plots
                        # plot model architecture to file
                        tf.keras.utils.plot_model(agent.model, to_file=os.path.join(exp_id, model_arch_file_name), show_shapes=True,
                                               show_layer_names=True)
                        print("Saved model architecture to file")
                    elif hasattr(agent, 'actor_model') and hasattr(agent, 'critic_model'):
                        actor_model_arch_file_name = '{}_{}_actor_model_architecture.png'.format(rl_variant, rl_agent)
                        tf.keras.utils.plot_model(agent.actor_model, to_file=os.path.join(exp_id, actor_model_arch_file_name), show_shapes=True,
                                               show_layer_names=True)
                        critic_model_arch_file_name = '{}_{}_critic_model_architecture.png'.format(rl_variant, rl_agent)
                        tf.keras.utils.plot_model(agent.critic_model, to_file=os.path.join(exp_id, critic_model_arch_file_name), show_shapes=True,
                                               show_layer_names=True)
                        print("Saved model architecture to file")
                    else:
                        print("Skipped saving model architecture due to model not found")
                except:
                    print("Skipped saving model architecture due to error")

                if not disable_only_training_code:

                    # Save model weights to file
                    # Save metrics, plots, models, etc.
                    try:
                        # save under run_dir
                        agent.save_model(os.path.join(run_dir,
                                                      f'model_{rl_agent}_{rl_variant}_{rerun_id}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.h5'))
                        print("Saved model to file")
                    except:
                        print("Skipped saving model due to error")

                    exp_num = exp_id.split("_")[1]
                    helper.plot_smoothed_rewards(agent.episode_reward_history, run_dir, smoothing_method="EMA",
                                    smoothing_level=0.9, save=True, prefix_str=f"run_{rerun_id}", save_plot_name=
                        f'{rl_variant}_{rl_agent}_run_{rerun_id}_rewards_{datetime.now().strftime("%Y%m%d-%H%M%S")}.png',
                        plot_title=f"{rl_variant} {rl_agent} {exp_id} run_{rerun_id} rewards")

                    print("Saved {} reward plot to file".format(exp_id))

                    # Log timings and other information
                    timings = {
                        'overall_experiment_time': time.time() - start_time,
                        'training_time': training_end_time - training_start_time
                        # 'inference_time': inference_end_time - inference_start_time,
                    }
                    print("Overall Experiment time: ", timings['overall_experiment_time'])
                    print("Training time: ", timings['training_time'])
                    # print("Inference time: ", timings['inference_time'])

                    #time per iteration

                    experiment_outcomes = {"experiment_config": exp_config,
                                           "other_parameters": agent.config_params,
                                           "model_details": {
                                               "input_shape": agent.input_shape,
                                                "output_shape": agent.output_shape,
                                               "trainable_params": agent.trainable_params,
                                           },
                                          "metrics":
                                                {"episode_reward_history": agent.episode_reward_history,
                                                 "episode_length_history": agent.episode_length_history,
                                                 "episode_count": agent.episode_count,
                                                 "average_timesteps_per_episode": agent.average_timesteps_per_episode,
                                                 "episode_rewards_min": agent.episode_rewards_min,
                                                 "episode_rewards_max": agent.episode_rewards_max,
                                                 "episode_rewards_mean": agent.episode_rewards_mean,
                                                 "episode_rewards_median": agent.episode_rewards_median,
                                                 "episode_rewards_std": agent.episode_rewards_std,
                                                    "episode_rewards_iqr": agent.episode_rewards_iqr,
                                                    "episode_rewards_q1": agent.episode_rewards_q1,
                                                    "episode_rewards_q3": agent.episode_rewards_q3,
                                                 "training_time": timings['training_time']
                                                 },
                                        "timings": timings
                                        }

                    # Log all experiment details
                    with open(os.path.join(run_dir, '{}_outcomes_run_{}_{}.json'.format(exp_id, rerun_id, datetime.now().strftime("%Y%m%d-%H%M%S"))), 'w') as file:
                        json.dump(experiment_outcomes, file, indent=4, cls=NumpyEncoder)

                    print(f"Experiment {exp_id} completed and results saved in {exp_id} folder")

        print(f"All experiments completed and results saved in respective experiment_folders: {experiment_ids}")

        # rewards_history = experiment_outcomes["metrics"]["episode_reward_history"]
        # helper.plot_smoothed_rewards(rewards_history, exp_id, rerun_id, smoothing_method="EMA", smoothing_level=0.9)
        # helper.plot_smoothed_rewards(rewards_history, exp_id, rerun_id, smoothing_method="SG_filter", smoothing_level=21)

        # Aggregate results from all reruns of an experiment
        base_dir = os.getcwd()
        print("base_dir: ", base_dir)
        subdirs_with_levels = [{'run': -1, exp_id: -2}]
        print("subdirs_with_levels: ", subdirs_with_levels)
        substring = 'outcome'
        extension = '.json'

        outcome_files = helper.filter_files_with_subdir_levels(base_dir, subdirs_with_levels, substring, extension)
        print(len(outcome_files))
        print("filtered_outcome_files: ", outcome_files)

        experiment_outcomes_list = []
        exp_rewards_history_list = []
        for outcome_file in outcome_files:
            with open(outcome_file, 'r') as file:
                exp_outcome = json.load(file)
                exp_rewards_history = exp_outcome["metrics"]["episode_reward_history"]
                exp_rewards_history_list.append(exp_rewards_history)
                experiment_outcomes_list.append(exp_outcome)

        helper.plot_averaged_rewards(exp_rewards_history_list, exp_id, save=True,
                plot_title=f"{exp_config['rl_variant']} {exp_config['rl_agent']} averaged rewards")

        averaged_reward_history = helper.compute_averaged_rewards(exp_rewards_history_list, method='plateau', fill_value=None)
        helper.plot_smoothed_rewards(averaged_reward_history, exp_id, smoothing_method="EMA",
                                     smoothing_level=0.9, save=True, prefix_str="averaged",
                        plot_title=f"{exp_config['rl_variant']} {exp_config['rl_agent']} averaged rewards")

        combined_outcome_dict = helper.combine_experiment_outcomes(experiment_outcomes_list, exp_id, combining_strategy = "mean")
        # helper.plot_smoothed_rewards(combined_outcome_dict["metrics"]["episode_reward_history"], exp_id, rerun_id=0, smoothing_method="EMA",
        #                              smoothing_level=0.9, save=False, pefix_str="averaged")


if __name__ == '__main__':
    # Load experiment configurations
    with open('Classical_Quantum_RL_experiments.json', 'r') as file:
        experiments_config = json.load(file)

    config_version = experiments_config['version']
    baseline_rl_experiments = experiments_config["experiments"][0]['baseline_rl_experiments']
    # policy_gradient_qpc_architecture_experiments = experiments_config["experiments"][1]['policy_gradient_qpc_experiments']
    # deep_q_learning_qpc_architecture_experiments = experiments_config["experiments"][2]['deep_q_learning_qpc_experiments']
    # actor_critic_qpc_architecture_experiments = experiments_config["experiments"][3]['actor_critic_qpc_experiments']

    # Experiments to run
    selected_experiments = baseline_rl_experiments
    total_experiments = len(selected_experiments)
    print(f"Total selected experiments: {total_experiments}")
    print("Baseline RL Experiments", selected_experiments)

    experiment_ids_to_skip = []#["experiment_1"]

    experiment_ids_to_run = ["experiment_3"] #, "experiment_3", "experiment_5"]

    print("Running experiments...")
    experiments_to_run = {exp_id: exp_config for exp_id, exp_config in selected_experiments.items()
                          if (exp_id not in experiment_ids_to_skip and
                              (exp_id in experiment_ids_to_run if experiment_ids_to_run else True))}
    print("Final experiments to run", experiments_to_run)

    run_experiments(experiments_to_run, reruns_per_experiment=1, start_from_rerun_id=4, skip_whole_training_block=False,
                    disable_only_training_code=False)



#=======================================================================================================================
#
# experiment_outcomes = {"experiment_config": exp_config,
#                        "metrics":
#                            {"episode_reward_history": agent.episode_reward_history,
#                             "episode_length_history": agent.episode_length_history,
#                             "actor_loss_history": agent.actor_loss_history,
#                             "critic_loss_history": agent.critic_loss_history,
#                             "episode_rewards_min": agent.episode_rewards_min,
#                             "episode_rewards_max": agent.episode_rewards_max,
#                             "episode_rewards_mean": agent.episode_rewards_mean,
#                             "episode_rewards_std": agent.episode_rewards_std,
#                             "env_metrics": {"pole_angle_history": agent.pole_angle_history,
#                                             "pole_angular_velocity_history": agent.pole_angular_velocity_history,
#                                             "cart_position_history": agent.cart_position_history,
#                                             "cart_velocity_history": agent.cart_velocity_history,
#                                             "action_distribution": agent.action_distribution,
#                                             "termination_reasons": agent.termination_reasons},
#
#                             "other_metrics": {"entropy_history": agent.entropy_history,
#                                               "advantage_values_history": agent.advantage_values_history,
#                                               "gradient_magnitudes_history": agent.gradient_magnitudes_history}
#                             },
#                        "timings": timings
#                        }
#=======================================================================================================================