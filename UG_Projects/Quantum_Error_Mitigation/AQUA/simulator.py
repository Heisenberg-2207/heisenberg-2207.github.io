import numpy as np
import time
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, thermal_relaxation_error
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService

class QuantumCircuitSimulator:
    """
    A class to simulate a quantum circuit with both ideal (statevector) and noisy simulations.
    
    Attributes:
        circuit (QuantumCircuit): Input quantum circuit.
        n_qubits (int): Number of qubits in the circuit.
        shots (int): Number of shots for noisy simulation.
        iterations (int): Number of simulation iterations.
        noise_params (tuple): (p1, p2) error rates for single- and two-qubit depolarizing errors.
    """
    def __init__(self, circuit, n_qubits, shots=8192, iterations=20, use_ibm=False):
        self.circuit = circuit.copy()
        self.n_qubits = n_qubits
        self.shots = shots
        self.iterations = iterations

        # Set up the ideal simulator (using statevector method)
        self.ideal_sim = AerSimulator(method="statevector")
        if 'GPU' in self.ideal_sim.available_devices():
            print("Using GPU for statevector simulation")
            self.ideal_sim.set_options(device='GPU')
        else:
            print("Using CPU for statevector simulation")
        
        # Create a noise model and set up the noisy simulator
        self.noise_model = self.get_ibm_noise_model() if use_ibm else self.create_noise_model()
        self.noisy_sim = AerSimulator(noise_model=self.noise_model)
    
    @staticmethod
    def get_ibm_noise_model(self):
        """Retrieve the noise model from an IBM quantum backend."""
        try:
            service = QiskitRuntimeService(name="Aniket")
            backend = service.backend("ibm_brisbane")
            return NoiseModel.from_backend(backend)
        except Exception as e:
            print(f"Error loading IBM noise model: {e}\nUsing fallback noise model...")
            return self.create_noise_model()

    @staticmethod
    def create_noise_model():
        """Create a noise model with depolarizing, readout, and thermal relaxation errors."""
        noise_model = NoiseModel()
        p_depol_1q, p_depol_2q, p_readout = 0.001, 0.01, 0.02
        T1, T2 = 50000, 70000  # Set T1 and T2 times in nanoseconds
        time_gates = {
            'u1': 0, 'u2': 50, 'u3': 100, 'rx': 50, 'ry': 50, 'rz': 0,
            'h': 50, 'x': 50, 'y': 50, 'z': 0, 'cx': 300, 'cz': 300,
            'reset': 1000, 'measure': 1000
        }
        error_1q = depolarizing_error(p_depol_1q, 1)
        error_2q = depolarizing_error(p_depol_2q, 2)
        error_readout = ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
        
        for gate, t in time_gates.items():
            if t <= 0:
                continue
            if gate in ['cx', 'cz']:
                error_thermal = thermal_relaxation_error(T1, T2, t).compose(thermal_relaxation_error(T1, T2, t))
                noise_model.add_all_qubit_quantum_error(error_thermal.compose(error_2q), [gate])
            elif gate not in ['reset', 'measure']:
                noise_model.add_all_qubit_quantum_error(thermal_relaxation_error(T1, T2, t).compose(error_1q), [gate])
        
        noise_model.add_all_qubit_readout_error(error_readout)
        
        if 'reset' in time_gates:
            noise_model.add_all_qubit_quantum_error(thermal_relaxation_error(T1, T2, time_gates['reset']), 'reset')
        
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
    
    def compile_circuits(self):
        """
        Compile two versions of the circuit:
          - Ideal circuit: saves the statevector.
          - Noisy circuit: includes measurement operations.
        
        Returns:
            tuple: (compiled_ideal, compiled_noisy)
        """
        # Ideal version: add statevector save instruction
        circuit_ideal = self.circuit.copy()
        circuit_ideal.save_statevector(label="sv")
        # Noisy version: add measurements to all qubits
        circuit_noisy = self.circuit.copy()
        circuit_noisy.measure_all()
        
        compiled_ideal = transpile(circuit_ideal, backend=self.ideal_sim, optimization_level=0)
        compiled_noisy = transpile(circuit_noisy, backend=self.noisy_sim, optimization_level=0)
        return compiled_ideal, compiled_noisy
    
    def run_simulation(self, verbose=False):
        """
        Run the ideal and noisy simulations for a number of iterations.
        
        Returns:
            tuple: (compiled_ideal, compiled_noisy, ideal_expectations, noisy_expectations)
                - compiled_ideal: The compiled circuit for ideal simulation.
                - compiled_noisy: The compiled circuit for noisy simulation.
                - ideal_expectations: Array of ideal <Z> expectation values (iterations x n_qubits).
                - noisy_expectations: Array of noisy <Z> expectation values (iterations x n_qubits).
        """
        compiled_ideal, compiled_noisy = self.compile_circuits()
        ideal_results = []
        noisy_results = []
        if verbose:
            print(f"Running {self.iterations} iterations...")
        start_time = time.time()
        for i in range(self.iterations):
            # Ideal simulation: run circuit and compute <Z> from statevector
            result_ideal = self.ideal_sim.run(compiled_ideal).result()
            statevector = result_ideal.data()["sv"]
            ideal_exp = self.compute_expectation_from_statevector(statevector, self.n_qubits)
            ideal_results.append(ideal_exp)
            
            # Noisy simulation: run circuit with shots and compute <Z> from counts
            result_noisy = self.noisy_sim.run(compiled_noisy, shots=self.shots).result()
            counts_noisy = result_noisy.get_counts()
            noisy_exp = self.compute_expectation_from_counts(counts_noisy, self.n_qubits)
            noisy_results.append(noisy_exp)
        elapsed = time.time() - start_time
        if verbose:
            print(f"Simulations completed in {elapsed:.2f} seconds")

        
        ideal_results = np.array(ideal_results)
        noisy_results = np.array(noisy_results)

        # Unpacking in case of single iteration
        if self.iterations == 1:
            # Remove any single-dimensional entries.
            ideal_results = np.squeeze(ideal_results, axis=0)
            noisy_results = np.squeeze(noisy_results, axis=0)

        return compiled_ideal, compiled_noisy, ideal_results.tolist(), noisy_results.tolist()
    
    def plot_expectations(self, ideal_expectations, noisy_expectations):
        """
        Plot the per-qubit Z expectation values over the simulation iterations.
        
        Parameters:
            ideal_expectations (np.array): Ideal <Z> expectation values (iterations x n_qubits).
            noisy_expectations (np.array): Noisy <Z> expectation values (iterations x n_qubits).
        """
        iterations_range = np.arange(ideal_expectations.shape[0])
        plt.figure(figsize=(14, 10))
        for q in range(self.n_qubits):
            plt.subplot(self.n_qubits, 1, q+1)
            plt.plot(iterations_range, ideal_expectations[:, q], "o-", label=f"Ideal Qubit {q}", color='blue')
            plt.plot(iterations_range, noisy_expectations[:, q], "s--", label=f"Noisy Qubit {q}", color='orange')
            plt.axhline(y=ideal_expectations[0, q], color='blue', linestyle='-', alpha=0.3)
            y_min = min(np.min(ideal_expectations[:, q]), np.min(noisy_expectations[:, q])) - 0.1
            y_max = max(np.max(ideal_expectations[:, q]), np.max(noisy_expectations[:, q])) + 0.1
            plt.ylim(y_min, y_max)
            plt.ylabel(f"<Z> Qubit {q}")
            plt.grid(True)
            plt.legend()
        plt.xlabel("Iteration")
        plt.suptitle(f"Per-Qubit Z Expectation Values over {ideal_expectations.shape[0]} Iterations\n{self.shots} shots per noisy iteration")
        plt.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    from qiskit.circuit.random import random_circuit
    # Generate a random circuit (without measurements)
    n_qubits = 4
    depth = 6
    circuit = random_circuit(n_qubits, depth, measure=False, seed=42)
    
    # Initialize the simulator with the given circuit and parameters
    simulator = QuantumCircuitSimulator(circuit, n_qubits, shots=8192, iterations=20)
    
    # Run the simulation to obtain compiled circuits and expectation values
    ideal_circuit, noisy_circuit, ideal_expectation, noisy_expectation = simulator.run_simulation()
    
    # Display results from the first iteration for demonstration
    print("\nIdeal Expectation Values (Iteration 1):", ideal_expectation[0])
    print("Noisy Expectation Values (Iteration 1):", noisy_expectation[0])
    
    # Plot the per-qubit expectation values over iterations
    simulator.plot_expectations(ideal_expectation, noisy_expectation)
