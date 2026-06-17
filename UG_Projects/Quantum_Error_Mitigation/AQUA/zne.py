import numpy as np
import matplotlib.pyplot as plt
import time

from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import Statevector

# MITIQ imports for ZNE
from mitiq.zne import execute_with_zne # type: ignore
from mitiq.zne.scaling import fold_gates_at_random # type: ignore
from mitiq.zne.inference import LinearFactory, RichardsonFactory, PolyExpFactory, AdaExpFactory # type: ignore

class ZNESimulator:
    """
    A class to simulate a quantum circuit under noise and perform
    Zero-Noise Extrapolation (ZNE) using MITIQ.
    
    It uses a statevector simulator (ideal) and a noisy simulator (with depolarizing errors)
    to compute per-qubit <Z> expectation values. It applies four different extrapolation
    methods: linear, Richardson, exponential, and polynomial.
    """
    
    def __init__(self, circuit, n_qubits, shots=8192, scale_factors=[1.0, 2.0, 3.0], noise_model=None):
        """
        Initialize the ZNESimulator.
        
        Parameters:
            circuit (QuantumCircuit): The input quantum circuit (without measurements).
            n_qubits (int): Number of qubits in the circuit.
            shots (int): Number of shots for noisy simulation.
            scale_factors (list): Noise scaling factors to use in ZNE.
            p1 (float): Depolarizing error rate for single-qubit gates.
            p2 (float): Depolarizing error rate for two-qubit gates.
        """
        self.circuit = circuit.copy()
        self.n_qubits = n_qubits
        self.shots = shots
        self.scale_factors = scale_factors

        # Ideal simulator: statevector simulation (noise-free)
        self.ideal_sim = AerSimulator(method="statevector")
        
        # Create noise model and noisy simulator
        self.noise_model = noise_model if noise_model is not None else self.create_noise_model()
        self.noisy_sim = AerSimulator(noise_model=self.noise_model)
        
    @staticmethod
    def create_noise_model(p1=0.01, p2=0.05):
        """
        Create a depolarizing noise model with the given error rates.
        """
        noise_model = NoiseModel()
        error_1 = depolarizing_error(p1, 1)
        error_2 = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cz', 'swap'])
        return noise_model

    @staticmethod
    def compute_expectation_from_statevector(statevector, n_qubits):
        """
        Compute per-qubit <Z> expectation values from a statevector.
        
        Parameters:
            statevector (Statevector or np.array): The final state of the circuit.
            n_qubits (int): Number of qubits.
        
        Returns:
            np.array: Array of <Z> expectation values for each qubit.
        """
        if isinstance(statevector, Statevector):
            statevector = statevector.data
        probabilities = np.abs(statevector)**2
        exp_vals = np.zeros(n_qubits)
        for q in range(n_qubits):
            prob_0 = 0.0
            prob_1 = 0.0
            for idx, prob in enumerate(probabilities):
                if prob > 0:
                    if (idx >> q) & 1:
                        prob_1 += prob
                    else:
                        prob_0 += prob
            exp_vals[q] = prob_0 - prob_1
        return exp_vals

    @staticmethod
    def compute_expectation_from_counts(counts, n_qubits):
        """
        Compute per-qubit <Z> expectation values from measurement counts.
        
        Parameters:
            counts (dict): Dictionary of measurement outcomes.
            n_qubits (int): Number of qubits.
        
        Returns:
            np.array: Array of <Z> expectation values for each qubit.
        """
        exp_vals = np.zeros(n_qubits)
        total_shots = sum(counts.values())
        for bitstring, count in counts.items():
            for q in range(n_qubits):
                # Qiskit uses little-endian for bitstrings
                if q < len(bitstring):
                    bit = bitstring[-1 - q]
                    value = 1 if bit == '0' else -1
                    exp_vals[q] += value * count
        return exp_vals / total_shots

    def run_ideal_simulation(self):
        """
        Run an ideal (noise-free) simulation using the statevector simulator.
        
        Returns:
            np.array: Ideal per-qubit <Z> expectation values.
        """
        circuit_ideal = self.circuit.copy()
        circuit_ideal.save_statevector(label="sv")
        compiled = transpile(circuit_ideal, backend=self.ideal_sim, optimization_level=0)
        result = self.ideal_sim.run(compiled).result()
        statevector = result.data()["sv"]
        return self.compute_expectation_from_statevector(statevector, self.n_qubits).tolist()

    def run_noisy_simulation(self):
        """
        Run a noisy simulation (with measurements) using the noisy simulator.
        
        Returns:
            np.array: Noisy per-qubit <Z> expectation values.
        """
        circuit_noisy = self.circuit.copy()
        circuit_noisy.measure_all()
        compiled = transpile(circuit_noisy, backend=self.noisy_sim, optimization_level=0)
        result = self.noisy_sim.run(compiled, shots=self.shots).result()
        counts = result.get_counts()
        return self.compute_expectation_from_counts(counts, self.n_qubits).tolist()

    def run_zne_simulation(self):
        """
        Run Zero-Noise Extrapolation (ZNE) using MITIQ's execute_with_zne.
        
        Applies four extrapolation methods: linear, Richardson, exponential, and polynomial.
        For each method, an executor function is defined per qubit that:
          - Adds measurements to the circuit,
          - Compiles and runs it on the noisy simulator,
          - Returns the scalar <Z> expectation for that qubit.
        
        Returns:
            dict: A dictionary mapping each extrapolation method to an np.array
                  of ZNE-mitigated per-qubit <Z> expectation values.
        """
        # Define factories for extrapolation.
        extrapolators = {
            "linear_zne": LinearFactory(scale_factors=self.scale_factors),
            "richardson_zne": RichardsonFactory(scale_factors=self.scale_factors),

            # TODO: Add them
            # "exponential_zne": PolyExpFactory(scale_factors=self.scale_factors),
            # "polynomial_zne": AdaExpFactory(scale_factors=self.scale_factors)
        }
        
        mitigated_results = {}  # key: method, value: np.array (n_qubits,)
        for method, factory in extrapolators.items():
            mitigated_vals = []
            for q in range(self.n_qubits):
                # Define an executor function for qubit q.
                def executor(circ, q=q):
                    circ_meas = circ.copy()
                    circ_meas.measure_all()
                    compiled = transpile(circ_meas, backend=self.noisy_sim, optimization_level=0)
                    result = self.noisy_sim.run(compiled, shots=self.shots).result()
                    counts = result.get_counts()
                    exp_vals = self.compute_expectation_from_counts(counts, self.n_qubits)
                    return exp_vals[q]
                # Run ZNE using MITIQ's execute_with_zne.
                mitigated_val = execute_with_zne(
                    self.circuit,
                    executor,
                    scale_noise=fold_gates_at_random,
                    factory=factory
                )
                mitigated_vals.append(mitigated_val)
            mitigated_results[method] = mitigated_vals
        return mitigated_results

    def run_simulation(self, return_ieal=False, return_noisy=False):
        """
        Run the full simulation: ideal, noisy, and ZNE-mitigated.
        
        Returns:
            tuple: (ideal_exps, noisy_exps, zne_results)
                - ideal_exps (np.array): Ideal per-qubit expectation values.
                - noisy_exps (np.array): Noisy per-qubit expectation values.
                - zne_results (dict): Dictionary mapping extrapolation method names
                                      to ZNE-mitigated per-qubit expectation values.
        """
        ideal_exps = self.run_ideal_simulation() if return_ieal else [-1]*self.n_qubits
        noisy_exps = self.run_noisy_simulation() if return_noisy else [-1]*self.n_qubits
        zne_results = self.run_zne_simulation()
        return ideal_exps, noisy_exps, zne_results

    def plot_results(self, ideal_exps, noisy_exps, zne_results):
        """
        Plot the ideal, noisy, and ZNE-mitigated per-qubit <Z> expectation values.
        """
        qubit_indices = np.arange(self.n_qubits)
        plt.figure(figsize=(10, 6))
        plt.plot(qubit_indices, ideal_exps, 'ko-', label="Ideal (Statevector)")
        plt.plot(qubit_indices, noisy_exps, 'rs--', label="Noisy")
        markers = {
            "linear_zne": 'o',
            "richardson_zne": 's',
            "exponential_zne": 'D',
            "polynomial_zne": '^'
        }
        linestyles = {
            "linear_zne": '-',
            "richardson_zne": '--',
            "exponential_zne": '-.',
            "polynomial_zne": ':'
        }
        for method, vals in zne_results.items():
            plt.plot(qubit_indices, vals, marker=markers[method], linestyle=linestyles[method],
                     label=f"ZNE ({method})")
        plt.xlabel("Qubit Index")
        plt.ylabel("<Z> Expectation Value")
        plt.title("Ideal vs Noisy vs ZNE Mitigated Expectation Values")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate a random circuit (without measurements).
    n_qubits = 5
    depth = 6
    shots = 8192
    scale_factors = [1.0, 2.0, 3.0]
    circuit = random_circuit(n_qubits, depth, measure=False, seed=42)
    
    # Initialize the ZNESimulator.
    zne_simulator = ZNESimulator(
        circuit=circuit,
        n_qubits=n_qubits,
        shots=shots,
        scale_factors=scale_factors,
        p1=0.01,
        p2=0.05
    )
    
    # Run the simulation.
    ideal_exps, noisy_exps, zne_results = zne_simulator.run_simulation()
    
    # Print the results.
    print("Ideal Expectation Values:", ideal_exps)
    print("Noisy Expectation Values:", noisy_exps)
    for method, vals in zne_results.items():
        print(f"ZNE Mitigated Values ({method}):", vals)
    
    # Plot the results.
    zne_simulator.plot_results(ideal_exps, noisy_exps, zne_results)
