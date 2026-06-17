import numpy as np
import matplotlib.pyplot as plt
import time
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, thermal_relaxation_error
from qiskit.quantum_info import Statevector

# MITIQ imports for ZNE
from mitiq.zne import execute_with_zne  # type: ignore
from mitiq.zne.scaling import fold_gates_at_random  # type: ignore
from mitiq.zne.inference import LinearFactory, RichardsonFactory  # type: ignore


class ZNESimulator:
    """
    A class to simulate a quantum circuit under noise and perform
    Zero-Noise Extrapolation (ZNE) using MITIQ.
    
    It uses a statevector simulator (ideal) and a noisy simulator (with depolarizing errors)
    to compute the overall <Z^⊗n> expectation value. It applies two extrapolation
    methods: linear and Richardson.
    """
    
    def __init__(self, n_qubits, shots=8192, scale_factors=[1.0, 2.0, 3.0], noise_model=None):
        """
        Initialize the ZNESimulator.
        
        Parameters:
            n_qubits (int): Number of qubits in the circuit.
            shots (int): Number of shots for noisy simulation.
            scale_factors (list): Noise scaling factors to use in ZNE.
        """
        self.circuit = None
        self.n_qubits = n_qubits
        self.shots = shots
        self.scale_factors = scale_factors

        # Ideal simulator: statevector simulation (noise-free)
        self.ideal_sim = AerSimulator(method="statevector")
        
        # Create noise model and noisy simulator.
        self.noise_model = noise_model if noise_model is not None else self.create_noise_model()
        self.noisy_sim = AerSimulator(noise_model=self.noise_model)
        
    @staticmethod
    def create_noise_model():
        """Create a noise model with depolarizing, readout, and thermal relaxation errors."""
        noise_model = NoiseModel()
        p_depol_1q, p_depol_2q, p_readout = 0.001, 0.01, 0.02
        T1, T2 = 50000, 70000  # in nanoseconds
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
        Compute the overall <Z^⊗n> expectation value from a statevector.
        
        Parameters:
            statevector (Statevector or np.array): The final state of the circuit.
            n_qubits (int): Number of qubits.
        
        Returns:
            float: Overall expectation value <Z^⊗n>.
        """
        # If statevector is a Qiskit Statevector object, extract its data.
        if hasattr(statevector, "data"):
            statevector = statevector.data
        probabilities = np.abs(statevector)**2
        expectation = 0.0
        for idx, prob in enumerate(probabilities):
            # Represent the basis state as an n-bit string.
            bitstring = format(idx, f'0{n_qubits}b')
            # Overall eigenvalue: product of +1 for '0' and -1 for '1'
            eigenvalue = np.prod([1 if bit == '0' else -1 for bit in bitstring])
            expectation += eigenvalue * prob
        return expectation

    @staticmethod
    def compute_expectation_from_counts(counts, n_qubits):
        """
        Compute the overall <Z^⊗n> expectation value from measurement counts.
        
        Parameters:
            counts (dict): Dictionary of measurement outcomes.
            n_qubits (int): Number of qubits.
        
        Returns:
            float: Overall expectation value <Z^⊗n>.
        """
        total_shots = sum(counts.values())
        expectation = 0.0
        for bitstring, count in counts.items():
            # Reverse bitstring if needed (Qiskit convention: little-endian).
            bitstring = bitstring[::-1]
            eigenvalue = np.prod([1 if bit == '0' else -1 for bit in bitstring])
            expectation += eigenvalue * count
        return expectation / total_shots

    def compile_circuits(self, circuit):
        """
        Compile two versions of the circuit:
          - Ideal circuit: saves the statevector.
          - Noisy circuit: includes measurement operations.
        
        Returns:
            tuple: (compiled_ideal, compiled_noisy)
        """
        self.circuit = circuit.copy()
        circuit_ideal = self.circuit.copy()
        circuit_ideal.save_statevector(label="sv")
        circuit_noisy = self.circuit.copy()
        circuit_noisy.measure_all()
        compiled_ideal = transpile(circuit_ideal, backend=self.ideal_sim, optimization_level=0)
        compiled_noisy = transpile(circuit_noisy, backend=self.noisy_sim, optimization_level=0)
        return compiled_ideal, compiled_noisy

    def run_simulation(self, circuit, run_ideal=True, run_noisy=True, run_zne=True, verbose=False):
        """
        Run the full simulation: ideal, noisy, and ZNE-mitigated (as specified by the arguments).

        Parameters
        ----------
        circuit : QuantumCircuit
            The input circuit to simulate.
        run_ideal : bool, optional
            Whether to run the ideal (statevector) simulation. Default is True.
        run_noisy : bool, optional
            Whether to run the noisy simulation (with measurements). Default is True.
        run_zne : bool, optional
            Whether to run Zero-Noise Extrapolation (ZNE). Default is True.
        verbose : bool, optional
            Whether to print verbose output. Default is False.

        Returns
        -------
        tuple:
            (ideal_exp, noisy_exp, zne_results)
                - ideal_exp (float or None): Overall ideal <Z^⊗n> expectation value,
                                            or None if run_ideal is False.
                - noisy_exp (float or None): Overall noisy <Z^⊗n> expectation value,
                                            or None if run_noisy is False.
                - zne_results (dict or None): Dictionary mapping extrapolation method names
                                            to ZNE-mitigated overall expectation values,
                                            or None if run_zne is False.
        """
        compiled_ideal, compiled_noisy = self.compile_circuits(circuit)
        
        ideal_exp = None
        noisy_exp = None
        zne_results = None

        # Run ideal simulation if requested.
        if run_ideal:
            result_ideal = self.ideal_sim.run(compiled_ideal).result()
            statevector = result_ideal.data()["sv"]
            ideal_exp = self.compute_expectation_from_statevector(statevector, self.n_qubits)
        
        # Run noisy simulation if requested.
        if run_noisy:
            result_noisy = self.noisy_sim.run(compiled_noisy, shots=self.shots).result()
            counts_noisy = result_noisy.get_counts()
            noisy_exp = self.compute_expectation_from_counts(counts_noisy, self.n_qubits)
        
        # Run ZNE if requested.
        if run_zne:
            extrapolators = {
                "linear_zne": LinearFactory(scale_factors=self.scale_factors),
                "richardson_zne": RichardsonFactory(scale_factors=self.scale_factors),
                # Additional factories can be added as needed.
            }
            zne_results = {}
            for method, factory in extrapolators.items():
                def executor(circ):
                    circ_meas = circ.copy()
                    circ_meas.measure_all()
                    compiled = transpile(circ_meas, backend=self.noisy_sim, optimization_level=0)
                    result = self.noisy_sim.run(compiled, shots=self.shots).result()
                    counts = result.get_counts()
                    return self.compute_expectation_from_counts(counts, self.n_qubits)
                mitigated_val = execute_with_zne(
                    circuit,
                    executor,
                    scale_noise=fold_gates_at_random,
                    factory=factory
                )
                zne_results[method] = mitigated_val
        
        if verbose:
            print("Simulation complete.")
        return ideal_exp, noisy_exp, zne_results


    def plot_results(self, ideal_exp, noisy_exp, zne_results):
        """
        Plot the overall <Z^⊗n> expectation values from ideal, noisy, and ZNE-mitigated simulations.
        """
        methods = ['Ideal', 'Noisy'] + list(zne_results.keys())
        values = [ideal_exp, noisy_exp] + [zne_results[m] for m in zne_results]
        
        plt.figure(figsize=(8, 6))
        plt.bar(methods, values, color=['black', 'red', 'blue', 'green'])
        plt.ylabel("<Z^⊗n> Expectation Value")
        plt.title("<Z^⊗n> Expectation Values for different Methods")
        plt.grid(axis="y")
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
        n_qubits=n_qubits,
        shots=shots,
        scale_factors=scale_factors
    )
    
    # Run the simulation (overall expectation calculations).
    ideal_exp, noisy_exp, zne_results = zne_simulator.run_simulation(circuit, verbose=True)
    
    print("\nIdeal <Z^⊗n> Expectation:", ideal_exp)
    print("Noisy <Z^⊗n> Expectation:", noisy_exp)
    for method, val in zne_results.items():
        print(f"ZNE Mitigated Value ({method}):", val)
    
    # Plot the overall expectation values.
    zne_simulator.plot_results(ideal_exp, noisy_exp, zne_results)
