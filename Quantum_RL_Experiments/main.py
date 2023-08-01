import json
import os
from helper import save_results, save_plots, save_model
from environments import create_environment

# Importing classical agents
from Quantum_RL_Experiments.classical_RL_agents.REINFORCE_classical import PolicyGradientClassicalAgent
from deep_q_learning_classical import DeepQLearningClassicalAgent
from actor_critic_classical import ActorCriticClassicalAgent

# Importing quantum agents
from policy_gradient_quantum import PolicyGradientQuantumAgent
from deep_q_learning_quantum import DeepQLearningQuantumAgent
from actor_critic_quantum import ActorCriticQuantumAgent

import json
import time
import os
import shutil
from datetime import datetime
from your_agent_module import ClassicalActorCritic, QuantumActorCritic  # Make sure to import your RL agents here
import matplotlib.pyplot as plt


def run_experiments(experiments_list):

    experiment_ids = [exp['experiment_id'] for exp in experiments_list]

    for experiment_id in experiment_ids:
        if not os.path.exists(experiment_id):
            ## shutil.rmtree(experiment_folder)
            os.makedirs(experiment_id)

    # Iterate over all the experiments in the config file
    for experiment_config in experiments_list:
        # Log experiment config
        experiment_id = experiment_config['experiment_id']
        with open(os.path.join(experiment_id, '{}_config.json'.format(experiment_id)), 'w') as file:
            json.dump(experiment_config, file, indent=4)

        start_time = time.time()

        # Unpack parameters
        rl_agent = experiment.get('rl_agent')                           # RL agent to use
        rl_variant = experiment.get('rl_variant')                       # Classical or Quantum
        exploration_strategy = experiment.get('exploration_strategy')   # Exploration strategy
        n_episodes = experiment.get('n_episodes')                       # Training episodes
        batch_size = experiment.get('batch_size')                       # Training batch size
        gamma = experiment.get('gamma')                                 # Discount factor
        n_qubits = experiment.get('n_qubits')                           # Quantum specific
        n_layers = experiment.get('n_layers')                           # Quantum specific
        qpc_architecture = experiment.get('qpc_architecture')           # Quantum specific


        # Initialize Agent
        agent = None
        if rl_variant == 'classical':
            if rl_agent == 'policy_gradient':
                agent = REINFORCEAgent(env_name='CartPole-v1', learning_rate=0.02, gamma=0.99, batch_size=16, n_episodes=2000)
            elif rl_agent == 'deep_q_learning':
                agent = DeepQLearningClassical(n_actions=2, gamma=0.99, n_episodes=2000, batch_size=16)
            elif rl_agent == 'actor_critic':
                agent = ActorCriticClassical(gamma=0.99, n_episodes=2000, max_steps_per_episode=10000, learning_rate=0.01)
        elif rl_variant == 'quantum':
            if rl_agent == 'policy_gradient':
                agent = PolicyGradientQuantum(env_name="CartPole-v1", n_qubits=4, n_layers=5, n_actions=2, gamma=1, n_episodes=1000, batch_size=10)
            elif rl_agent == 'deep_q_learning':
                agent = DeepQLearningQuantum(n_qubits=4, n_layers=5, n_actions=2, gamma=0.99, n_episodes=2000, batch_size=16)
            elif rl_agent == 'actor_critic':
                agent = ActorCriticQuantum(gamma=0.99, n_episodes=2000, max_steps_per_episode=10000, learning_rate=0.01)


        # Training
        training_start_time = time.time()
        agent.train()
        training_end_time = time.time()

        # Inference
        inference_start_time = time.time()
        agent.play()
        inference_end_time = time.time()

        # Save metrics, plots, models, etc.
        agent.save_model(os.path.join(experiment_id, f'model_{rl_agent}_{rl_variant}_{time.time()}.h5'))
        plt.plot(agent.episode_reward_history)
        plt.savefig(os.path.join(experiment_folder, f'reward_plot_{rl_agent}_{rl_variant}_{time.time()}.png'))

        # Log timings and other information
        timings = {
            'overall_time': time.time() - start_time,
            'training_time': training_end_time - training_start_time,
            'inference_time': inference_end_time - inference_start_time,
        }

        experiment_outcomes = {"experiment_config": experiment_config,
                              "metrics": {"episode_reward_history": agent.episode_reward_history,
                                            "episode_length_history": agent.episode_length_history,
                                            "episode_loss_history": agent.episode_loss_history,
                                            "episode_reward_mean": agent.episode_reward_mean,
                                            "episode_length_mean": agent.episode_length_mean,
                                            "episode_loss_mean": agent.episode_loss_mean
                                          },
                              "timings": timings
                                }


        # Log all experiment details
        with open(os.path.join(experiment_id, '{}_outcomes.json'.format(experiment_id)), 'w') as file:
            json.dump(experiment_outcomes, file, indent=4)

        print(f"Experiment {experiment_id} completed and results saved in {experiment_id} folder")

    print(f"All experiments completed and results saved in respective experiment_folders: {experiment_ids}")



    # Save the results, plots, and model
    save_results(results, experiment_outcomes)
    save_plots(metrics, experiment_config)
    save_model(model, experiment_config)

if __name__ == '__main__':
    # Load experiment configurations
    with open('Classical_Quantum_RL_experiments.json', 'r') as file:
        experiments_config = json.load(file)

    # Create directories for results, plots, and models
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    config_version = config['version']
    baseline_rl_experiments = experiments_config["experiments"]['baseline_rl_experiments']
    policy_gradient_n_layer_experiments = experiments_config["experiments"]['policy_gradient_n_layer_experiments']
    policy_gradient_qpc_architecture_experiments = experiments_config["experiments"]['policy_gradient_qpc_architecture_experiments']
    deep_q_learning_n_layer_experiments = experiments_config["experiments"]['deep_q_learning_n_layer_experiments']
    deep_q_learning_qpc_architecture_experiments = experiments_config["experiments"]['deep_q_learning_qpc_architecture_experiments']
    actor_critic_n_layer_experiments = experiments_config["experiments"]['actor_critic_n_layer_experiments']
    actor_critic_qpc_architecture_experiments = experiments_config["experiments"]['actor_critic_qpc_architecture_experiments']

    run_experiments(baseline_rl_experiments)
