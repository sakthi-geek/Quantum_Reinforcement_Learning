
# Update package resources to account for version changes.
import importlib, pkg_resources
importlib.reload(pkg_resources)

import tensorflow as tf
import tensorflow_quantum as tfq

import gym, cirq, sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
tf.get_logger().setLevel('ERROR')

from Quantum_RL_Experiments.helper import ReUploadingPQC, Alternating





class PolicyGradientQuantum:
    def __init__(self, env_name = "CartPole-v1", seed=39, n_qubits=4, n_layers=5, n_actions=2, gamma=1, n_episodes=3000,
                 batch_size=16, state_bounds=np.array([2.4, 2.5, 0.21, 2.5]), learning_rate={"in": 0.1, "var":0.01, "out": 0.1}):

        # Environment -----------------------
        self.env_name = env_name
        self.seed = seed

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        # random.seed(self.seed)

        # Model ---------------------------------------
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions

        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)
        self.ops = [cirq.Z(q) for q in self.qubits]
        self.observables = [reduce((lambda x, y: x * y), self.ops)]  # Z_0*Z_1*Z_2*Z_3

        # # Check that this produces a circuit that is alternating between variational and encoding layers.
        # _n_qubits, _n_layers = 3, 1
        # qubits = cirq.GridQubit.rect(1, _n_qubits)
        # circuit, _, _ = self.generate_circuit(qubits, _n_layers)
        # SVGCircuit(circuit)

        self.in_lr = learning_rate["in"]
        self.var_lr = learning_rate["var"]
        self.out_lr = learning_rate["out"]
        # Prepare the optimizers
        self.optimizer_in = tf.keras.optimizers.Adam(learning_rate=self.in_lr, amsgrad=True)
        self.optimizer_var = tf.keras.optimizers.Adam(learning_rate=self.var_lr, amsgrad=True)
        self.optimizer_out = tf.keras.optimizers.Adam(learning_rate=self.out_lr, amsgrad=True)

        # Assign the model parameters to each optimizer
        self.w_in, self.w_var, self.w_out = 1, 0, 2

        self.model = self.generate_model_policy(self.qubits, self.n_layers, self.n_actions, 1.0, self.observables)
        tf.keras.utils.plot_model(self.model, show_shapes=True, dpi=70)

        self.input_shape = self.model.input_shape
        self.output_shape = self.model.output_shape
        self.trainable_params = self.model.count_params()
        print(self.input_shape, self.output_shape, self.trainable_params)

        # Configuration parameters----------------------
        self.state_bounds = state_bounds
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.learning_rate = learning_rate
        self.in_lr = learning_rate["in"]
        self.var_lr = learning_rate["var"]
        self.out_lr = learning_rate["out"]

        self.config_params = {
            "num_obsevables": len(self.observables),
            "qubits": str(self.qubits),
            "ops": str(self.ops),
            "observables": str(self.observables),
        }

        # print all parameters
        print("----------------------")
        print("Environment: ", self.env_name)
        print("Seed: ", self.seed)
        print("Gamma: ", self.gamma)
        print("Number of Episodes: ", self.n_episodes)
        print("batch Size: ", self.batch_size)
        print("Learning Rate: ", self.learning_rate)
        print("State Bounds: ", self.state_bounds)
        print("Number of Qubits: ", self.n_qubits)
        print("Number of Layers: ", self.n_layers)
        print("Number of Actions: ", self.n_actions)
        print("Number of Observables: ", len(self.observables))
        print("Qubits: ", self.qubits)
        print("Ops: ", self.ops)
        print("Observables: ", self.observables)
        print("Model: ", self.model.summary())
        print("Input Shape: ", self.input_shape)
        print("Output Shape: ", self.output_shape)
        print("Trainable Parameters: ", self.trainable_params)
        print("lr_schedule: ", None)
        print("----------------------")

        # Metrics ---------------------------
        self.episode_reward_history = []
        self.episode_length_history = []



    def one_qubit_rotation(self, qubit, symbols):
        """
        Returns Cirq gates that apply a rotation of the bloch sphere about the X,
        Y and Z axis, specified by the values in `symbols`.
        """
        return [cirq.rx(symbols[0])(qubit),
                cirq.ry(symbols[1])(qubit),
                cirq.rz(symbols[2])(qubit)]

    def entangling_layer(self, qubits):
        """
        Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
        """
        cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
        cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
        return cz_ops

    def generate_circuit(self, qubits, n_layers):
        """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
        # Number of qubits
        n_qubits = len(qubits)

        # Sympy symbols for variational angles
        params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
        params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

        # Sympy symbols for encoding angles
        inputs = sympy.symbols(f'x(0:{n_layers})'+f'_(0:{n_qubits})')
        inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

        # Define circuit
        circuit = cirq.Circuit()
        for l in range(n_layers):
            # Variational layer
            circuit += cirq.Circuit(self.one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
            circuit += self.entangling_layer(qubits)
            # Encoding layer
            circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

        # Last varitional layer
        circuit += cirq.Circuit(self.one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))

        return circuit, list(params.flat), list(inputs.flat)


    def generate_model_policy(self, qubits, n_layers, n_actions, beta, observables):
        """Generates a Keras model for a data re-uploading PQC policy."""

        input_tensor = tf.keras.Input(shape=(len(qubits), ), dtype=tf.dtypes.float32, name='input')
        re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables)([input_tensor])
        process = tf.keras.Sequential([
            Alternating(n_actions),
            tf.keras.layers.Lambda(lambda x: x * beta),
            tf.keras.layers.Softmax()
        ], name="observables-policy")
        policy = process(re_uploading_pqc)
        model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

        return model

    def gather_episodes(self, state_bounds, n_actions, model, n_episodes, env_name):
        """Interact with environment in batched fashion."""

        trajectories = [defaultdict(list) for _ in range(n_episodes)]
        envs = [gym.make(env_name) for _ in range(n_episodes)]

        # Set seed for each environment instance for reproducibility
        seeded_envs = []
        if self.seed is not None:
            for i, env in enumerate(envs):
                env.seed(self.seed + i)
                seeded_envs.append(env)

        done = [False for _ in range(n_episodes)]
        states = [e.reset() for e in seeded_envs]

        while not all(done):
            unfinished_ids = [i for i in range(n_episodes) if not done[i]]
            normalized_states = [s / state_bounds for i, s in enumerate(states) if not done[i]]

            for i, state in zip(unfinished_ids, normalized_states):
                trajectories[i]['states'].append(state)

            # Compute policy for all unfinished envs in parallel
            states = tf.convert_to_tensor(normalized_states)
            action_probs = model([states])

            # Store action and transition all environments to the next state
            states = [None for i in range(n_episodes)]
            for i, policy in zip(unfinished_ids, action_probs.numpy()):
                action = np.random.choice(n_actions, p=policy)
                states[i], reward, done[i], _ = seeded_envs[i].step(action)
                trajectories[i]['actions'].append(action)
                trajectories[i]['rewards'].append(reward)

        return trajectories

    # Implement a function that updates the policy using states, actions and returns:
    @tf.function
    def reinforce_update(self, states, actions, returns, model):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            logits = model(states)
            p_actions = tf.gather_nd(logits, actions)
            log_probs = tf.math.log(p_actions)
            loss = tf.math.reduce_sum(-log_probs * returns) / self.batch_size
        grads = tape.gradient(loss, model.trainable_variables)
        for optimizer, w in zip([self.optimizer_in, self.optimizer_var, self.optimizer_out], [self.w_in, self.w_var, self.w_out]):
            optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])


    def compute_returns(self, rewards_history, gamma):
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


    def train(self):
        """Train the agent."""
        print(f"Training for {self.n_episodes //self.batch_size} batches.")
        # Start training the agent
        episode_reward_history = []
        for batch in range(self.n_episodes // self.batch_size):
            # Gather episodes
            episodes = self.gather_episodes(self.state_bounds, self.n_actions, self.model, self.batch_size, self.env_name)

            # Group states, actions and returns in numpy arrays
            states = np.concatenate([ep['states'] for ep in episodes])
            actions = np.concatenate([ep['actions'] for ep in episodes])
            rewards = [ep['rewards'] for ep in episodes]
            returns = np.concatenate([self.compute_returns(ep_rwds, self.gamma) for ep_rwds in rewards])
            returns = np.array(returns, dtype=np.float32)

            id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

            # Update model parameters.
            self.reinforce_update(states, id_action_pairs, returns, self.model)

            # Store collected rewards
            for ep_rwds in rewards:
                self.episode_reward_history.append(np.sum(ep_rwds))
                self.episode_length_history.append(len(ep_rwds))    # TODO: check if this is correct

            avg_rewards = np.mean(self.episode_reward_history[-10:])

            print('Finished episode', (batch + 1) * self.batch_size, 'Average rewards: ', avg_rewards)

            if avg_rewards >= 500.0:
                break

        self.episode_count = len(self.episode_reward_history)
        self.average_timesteps_per_episode = np.mean(self.episode_length_history)
        self.episode_rewards_min = np.min(self.episode_reward_history)
        self.episode_rewards_max = np.max(self.episode_reward_history)
        self.episode_rewards_mean = np.mean(self.episode_reward_history)
        self.episode_rewards_median = np.median(self.episode_reward_history)
        self.episode_rewards_std = np.std(self.episode_reward_history)
        self.episode_rewards_iqr = np.subtract(*np.percentile(self.episode_reward_history, [75, 25]))
        self.episode_rewards_q1 = np.percentile(self.episode_reward_history, 25)
        self.episode_rewards_q3 = np.percentile(self.episode_reward_history, 75)

        self.plot_rewards()



    def save_model(self, path):
        """Save the model to the specified path."""
        tf.keras.save_model(self.model, path)

    def load_model(self, path):
        """Load the model from the specified path."""
        self.model = tf.keras.load_model(path)

    def plot_rewards(self):
        """Plot the reward history of the agent."""
        plt.figure(figsize=(10,5))
        plt.plot(self.episode_reward_history)
        plt.xlabel('Episode')
        plt.ylabel('Collected rewards')
        plt.show()

