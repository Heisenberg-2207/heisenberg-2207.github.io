from util import *
from simulator import QuantumCircuitSimulator
from converter import CircuitConverter
from zne import ZNESimulator
from generator import QuantumGenerator
from extractor import QuantumGraphExtractor

from qiskit_aer import AerSimulator

# Create an instance of QuantumGenerator with desired parameters.
qgen = QuantumGenerator(
    n_qubits=2,
    depth=(10, 11, 2),         # Depth values: 4, 6, 8
    circuits_per_depth=2,
    shots=8192,
    scale_factors=[1.0, 2.0, 3.0],
    observable_mode="rand",   # Use random observables ("X", "Y", or "Z")
    fixed_pauli=None,         # If fixed mode is used, provide a string e.g. "XYZZZ"
    optimization_level=0,
    transpile_backend=AerSimulator(),   # Uses default GenericBackendV2 if not provided.
    conversion_type='graph',  # Options: 'graph', 'freq', 'cnn' etc.
    save=True,
    filename="data/raw_data_graph_q20_d10"
)

# Generate the data and store it in a DataFrame.
df = qgen.generate_data(output_format='csv')
print("Generated Data")

backend = qgen.transpile_backend  # Replace with actual backend if available.
extractor = QuantumGraphExtractor(filepath='data/raw_data_graph_q20_d10.csv', n_qubits=5, backend=backend, data_type="graph")
extractor.process()
data_objects = extractor.get_data()
if data_objects is not None:
    print("Data processing complete.")
extractor.export_data("data/processed_data_graph_q2_d10", output_format="csv")


