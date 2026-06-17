from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.providers.fake_provider import GenericBackendV2
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

# Import your custom modules.
from simulator import QuantumCircuitSimulator
from converter import CircuitConverter
from zne import ZNESimulator


class QuantumGenerator:
    """
    QuantumGenerator generates a dataset of quantum circuit simulations.
    
    For each circuit depth in a specified range, it:
      - Generates a random circuit.
      - Adds measurement observables based on either a random or fixed Pauli pattern.
      - Transpiles the circuit using a chosen backend.
      - Runs an ideal (statevector) simulation and a noisy simulation (with a noise model)
        using QuantumCircuitSimulator.
      - Converts the circuit using CircuitConverter.
      - Runs Zero-Noise Extrapolation (ZNE) using ZNESimulator.
      - Stores the results (observable pattern, circuits, expectation values, converted dataset, ZNE results).
      - Optionally saves the results to a CSV file.
    """

    def __init__(self, n_qubits, depth=(4, 10, 2), circuits_per_depth=2, shots=8192,
                 scale_factors=[1.0, 2.0, 3.0], observable_mode="rand", fixed_pauli=None,
                 optimization_level=0, transpile_backend=None, conversion_type=['graph'],
                 save=False, filename="raw_data"):
        """
        Initialize QuantumGenerator with experimental parameters.
        
        Parameters:
            n_qubits (int): Number of qubits.
            depth (tuple): (start, end, step) parameters for circuit depths.
            circuits_per_depth (int): Number of circuits per depth.
            shots (int): Number of shots for simulation.
            scale_factors (list): Noise scale factors for ZNE.
            observable_mode (str): "rand" for random observables, "fixed" for fixed pattern.
            fixed_pauli (str): If "fixed", the Pauli string (e.g. "XYZZZ") to use.
            optimization_level (int): Optimization level for transpilation.
            transpile_backend: Backend used for transpiling (default: GenericBackendV2).
            conversion_type (str): Type of conversion ("graph" or other types).
            save (bool): Whether to save the results as a CSV/JSON/PKL file.
            filename (str): Filename for the file to be saved.
        """
        self.n_qubits = n_qubits
        self.depth_params = depth  # (start, end, step)
        self.circuits_per_depth = circuits_per_depth
        self.shots = shots
        self.scale_factors = scale_factors
        self.observable_mode = observable_mode.lower()
        self.fixed_pauli = fixed_pauli
        self.optim_lvl = optimization_level
        self.conversion_type = conversion_type # List of conversions
        self.num_conversion = len(conversion_type)
        self.save = save
        self.filename = filename

        # Use provided transpile backend or default to a generic backend.
        self.transpile_backend = transpile_backend if transpile_backend is not None else GenericBackendV2(self.n_qubits)

        # Classes init for the pipeline
        self.simulator = QuantumCircuitSimulator(n_qubits=self.n_qubits, shots=self.shots, iterations=1, use_ibm=False)
        self.zne_simulator = ZNESimulator(n_qubits=self.n_qubits, shots=self.shots,
                                          scale_factors=self.scale_factors, noise_model=self.simulator.noise_model)
        self.converter = CircuitConverter(max_gate_attributes=3, bin_min=-3.14, bin_max=3.14, bin_size=0.5, mode='bin')

        # Data container to store results for each depth.
        self.data = {conv_type: [] for conv_type in self.conversion_type}

    def pauli_observable(self, circ):
        """
        Add observable rotations to the circuit based on a chosen Pauli pattern.
        
        For each qubit:
          - "X": apply Hadamard (H) to rotate measurement basis.
          - "Y": apply Sdg then H.
          - "Z": no rotation needed.
        
        Parameters:
            circ (QuantumCircuit): The input circuit.
            
        Returns:
            tuple: Modified circuit and the Pauli string used.
        """
        # Choose the Pauli pattern.
        if self.observable_mode == "rand":
            pauli_str = "".join(np.random.choice(list("XYZ"), size=self.n_qubits))
        else:
            pauli_str = self.fixed_pauli if self.fixed_pauli is not None else "Z" * self.n_qubits

        # Apply basis-changing rotations.
        for i, p in enumerate(pauli_str):
            if p == "X":
                circ.h(i)
            elif p == "Y":
                circ.sdg(i)
                circ.h(i)
            # For "Z", no rotation is required.
        return circ, pauli_str

    def generate_data(self, output_format="csv"):
        """
        Generate the simulation data over a range of circuit depths.
        
        Parameters:
            output_format (str): The format for saving the data. Options: "csv", "json", "pickle".
                
        Returns:
            pd.DataFrame: A DataFrame containing the generated data with proper column names.
        """
        # Iterate over specified depths.
        for depth in range(*self.depth_params):
            print(f"Generating circuits at depth: {depth}")
            # Generate a number of circuits at this depth.
            for _ in tqdm(range(self.circuits_per_depth), desc=f"Circuits at depth {depth}", leave=True):

                # Generate a random circuit (without measurements).
                entry = {"Depth": depth}

                circ = random_circuit(self.n_qubits, depth, measure=False)
                # entry["Circuit"] = circ

                # Apply measurement observables.
                circ, pauli_str = self.pauli_observable(circ)
                # Save the observable pattern.
                entry["ObservablePattern"] = pauli_str
                
                # Transpile the circuit.(No measurement yet)
                compiled_circ = transpile(circ, backend=self.transpile_backend, optimization_level=self.optim_lvl)
                # entry["CompiledCircuit"] = compiled_circ

                # Run simulations using QuantumCircuitSimulator.
                ideal_circuit, noisy_circuit, ideal_expectation, noisy_expectation = self.simulator.run_simulation(compiled_circ)
                # entry["IdealCircuit"] = ideal_circuit
                entry["NoisyCircuit"] = noisy_circuit # These have additional .measure_all(True)
                entry["IdealExpectation"] = ideal_expectation
                entry["NoisyExpectation"] = noisy_expectation

                self.circuit = compiled_circ
                # Run ZNE simulation using ZNESimulator, accepts parameters return_ieal and return_noisy.
                ideal_exps, noisy_exps, zne_results = self.zne_simulator.run_simulation(compiled_circ, run_ideal=False, run_noisy=False)
                entry["ZNEResults"] = zne_results

                # Convert the noisy circuit using multiple conversion types.
                for conv_type in self.conversion_type:
                    dataset_conv = self.converter.convert(noisy_circuit, conversion_type=conv_type)
                    entry_copy = entry.copy()  # Create a copy for each conversion type
                    entry_copy["ConvertedDataset"] = dataset_conv
                    self.data[conv_type].append(entry_copy)
        
        # Saving and returning DataFrames
        df_dict = {}
        for conv_type, data in self.data.items():
            df = pd.DataFrame(data, columns=["Depth", "ObservablePattern", "NoisyCircuit", # "Circuit", "CompiledCircuit", "IdealCircuit",
                                             "IdealExpectation", "NoisyExpectation", "ConvertedDataset", "ZNEResults"])
            df_dict[conv_type] = df
    
            if self.save:
                filename = f"{self.filename}_{conv_type}_{self.n_qubits}_{self.depth_params}_{self.circuits_per_depth}"
                fmt = output_format.lower()
                if fmt == "csv":
                    df.to_csv(filename + ".csv", index=False)
                elif fmt == "json":
                    df.to_json(filename + ".json", orient="records", lines=True)
                elif fmt == "pickle":
                    df.to_pickle(filename + ".pkl")
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")
    
        return df_dict



if __name__ == "__main__":
    # Create an instance of QuantumGenerator with desired parameters.
    qgen = QuantumGenerator(
        n_qubits=4,
        depth=(2, 3, 2),         # Depth values: 4, 6, 8
        circuits_per_depth=1000,
        shots=8192,
        scale_factors=[1.0, 2.0, 3.0],
        observable_mode="rand",   # Use random observables ("X", "Y", or "Z")
        fixed_pauli=None,         # If fixed mode is used, provide a string e.g. "XYZZZ"
        optimization_level=0,
        transpile_backend=None,   # Uses default GenericBackendV2 if not provided.
        conversion_type=["graph"],  # List of conversions, options: 'graph', 'freq', 'cnn' etc.
        save=True,
        filename="../../data/raw_data_graph" # Additional info like nqbit, depth_params, circuits_per_depth are appended to file-name
    )
    
    # Generate the data and store it in a DataFrame.
    df = qgen.generate_data()
    print("Generated Data:")
    print(df)

