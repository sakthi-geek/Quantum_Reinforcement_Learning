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

def run_experiment(experiment_config):
    # Retrieve parameters from the experiment config
    rl_variant = experiment_config['rl_variant']
    environment_name = experiment_config['environment']
    agent_type = experiment_config['agent_type']
    num_layers = experiment_config['num_layers']
    num_qubits = experiment_config['num_qubits']
    qpc_architecture_name = experiment_config['qpc_architecture_name']

    # Create environment
    env = create_environment(environment_name)

    # Create the appropriate RL agent
    if agent_type == 'policy_gradient':
        agent_class = PolicyGradientQuantumAgent if rl_variant == 'quantum' else PolicyGradientClassicalAgent
    elif agent_type == 'deep_q_learning':
        agent_class = DeepQLearningQuantumAgent if rl_variant == 'quantum' else DeepQLearningClassicalAgent
    elif agent_type == 'actor_critic':
        agent_class = ActorCriticQuantumAgent if rl_variant == 'quantum' else ActorCriticClassicalAgent

    agent = agent_class(env, num_layers, num_qubits, qpc_architecture_name)

    # Run the experiment
    results, metrics, model = agent.train()

    # Save the results, plots, and model
    save_results(results, experiment_config)
    save_plots(metrics, experiment_config)
    save_model(model, experiment_config)

if __name__ == '__main__':
    # Load experiment configurations
    with open('experiments.json', 'r') as file:
        experiment_configs = json.load(file)

    # Create directories for results, plots, and models
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Loop through and run all experiments
    for experiment_config in experiment_configs:
        run_experiment(experiment_config)
