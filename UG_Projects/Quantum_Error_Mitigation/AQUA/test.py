import os
os.system("pip install torch torch_geometric mitiq qiskit qiskit_aer qiskit_ibm_runtime ply")


from util import *
from simulator import QuantumCircuitSimulator
from converter import CircuitConverter
from zne import ZNESimulator
from generator import QuantumGenerator
from extractor import QuantumGraphExtractor

# Create an instance of QuantumGenerator with desired parameters.
qgen = QuantumGenerator(
    n_qubits=20,
    depth=(10, 11, 2),         # Depth values: 4, 6, 8
    circuits_per_depth=1000,
    shots=8192,
    scale_factors=[1.0, 2.0, 3.0],
    observable_mode="rand",   # Use random observables ("X", "Y", or "Z")
    fixed_pauli=None,         # If fixed mode is used, provide a string e.g. "XYZZZ"
    optimization_level=0,
    transpile_backend=None,   # Uses default GenericBackendV2 if not provided.
    conversion_type='graph',  # Options: 'graph', 'freq', etc.
    save=True,
    filename="data/raw_data_graph"
)

# Generate the data and store it in a DataFrame.
df = qgen.generate_data(output_format='csv')


# Load the dataset and convert columns from string to dictionary as needed.
backend = qgen.transpile_backend  # Replace with actual backend if available.
extractor = QuantumGraphExtractor(filepath='data/raw_data_graph.csv', n_qubits=qgen.n_qubits, backend=backend, data_type="graph")
extractor.process()
data_objects = extractor.get_data()
if data_objects is not None:
    print("Data processing complete.")
extractor.export_data("data/processed_data_graph", output_format="csv")
