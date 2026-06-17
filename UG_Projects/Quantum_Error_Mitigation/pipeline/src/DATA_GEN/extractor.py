import os
import ast
import csv
import json
import torch
import pandas as pd
from torch_geometric.data import Data
from qiskit.providers.fake_provider import GenericBackendV2

class QuantumGraphExtractor:
    """
    Extracts graph-based representations from a dataset of quantum circuits.

    The dataset must contain at least the following columns:
      - 'circuit_id': Unique identifier for each circuit.
      - 'Depth': Circuit depth.
      - 'ConvertedDataset': A dictionary (or its string representation) containing nodes and edges.
      - 'NoisyExpectation': A single noisy overall expectation value.
      - 'IdealExpectation': A single ideal overall expectation value.
      - 'ObservablePattern': A string representing the Pauli observable per qubit (e.g., "XYZ").
      - 'ZNEResults': A string representing a dictionary with error mitigation results.
    """

    def __init__(self, filepath, n_qubits, backend, data_type="graph"):
        """
        Initializes the QuantumGraphExtractor.

        Parameters:
            filepath (str): Path to the data file.
            n_qubits (int): Number of qubits in the circuits.
            backend: An object with an attribute `operation_names` (e.g., a quantum simulator backend).
            data_type (str): Type of conversion. Supported: "graph", "freq", and "cnn".
        """
        self.dataset = self.load_dataset(filepath=filepath)
        self.n_qubits = n_qubits
        self.data_list = []  # Will hold processed objects (PyG Data objects or dicts).
        self.data_type = data_type.lower()

        try:
            # Retrieve gate operation names from the backend and extend with additional types (only for graph).
            gate_ops = backend.operation_names  # e.g., ['h', 'cx', ...]
            extended_gate_ops = ["in"] + gate_ops + ["out", "barrier"]
            self.mapping = {gate: idx for idx, gate in enumerate(extended_gate_ops)}
            self.num_gate_types = len(self.mapping)
        except Exception as e:
            raise ValueError(f"Error initializing gate operations: {e}")

    @staticmethod
    def load_dataset(filepath):
        """
        Loads a dataset from the given filepath. Supports CSV, JSON, or Pickle formats.

        For CSV files, it applies converters to convert columns containing string
        representations of dictionaries/lists to their appropriate Python objects.

        Parameters:
            filepath (str): Path to the data file.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        try:
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()
            converters = {
                "ConvertedDataset": ast.literal_eval,
                "NoisyExpectation": ast.literal_eval,
                "IdealExpectation": ast.literal_eval,
                "ZNEResults": ast.literal_eval
            }
            if ext == ".csv":
                df = pd.read_csv(filepath, converters=converters)
            elif ext == ".json":
                df = pd.read_json(filepath, orient="records", lines=True)
            elif ext in [".pkl", ".pickle"]:
                df = pd.read_pickle(filepath)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
            return df
        except Exception as e:
            raise IOError(f"Error loading dataset from {filepath}: {e}")

    def convert_cnn(self):
        """
        Converts the DataFrame (assumed to contain CNN data) to a list of dictionaries.

        Now, the expectation values are assumed to be overall single values rather than per-qubit arrays.
        For each sample, a dictionary is built with keys:
            "cnn": Flattened CNN tensor (list),
            "in_channels": number of input channels,
            "n_total": number of rows,
            "depth": number of columns (time slices),
            "ideal": overall ideal expectation,
            "noisy": overall noisy expectation,
            "zne": overall ZNE expectation (if available).
        The resulting list is stored in self.data_list.
        """
        # Compute a global one-hot mapping and maximum parameter length across the dataset.
        unique_gates = set()
        max_attributes = 0
        for _, row in self.dataset.iterrows():
            conv_obj = row.get("ConvertedDataset", None)
            if conv_obj is None:
                raise ValueError("Missing 'ConvertedDataset' column in a row.")
            if isinstance(conv_obj, str):
                conv_dict = ast.literal_eval(conv_obj)
            else:
                conv_dict = conv_obj
            data_matrix = conv_dict.get("data", None)
            if data_matrix is None:
                raise ValueError("The ConvertedDataset dictionary must contain a 'data' key.")
            for row_entry in data_matrix:
                for cell in row_entry[1:]:
                    if isinstance(cell, list) and len(cell) >= 1:
                        unique_gates.add(cell[0])
                        param_len = len(cell) - 1
                        max_attributes = max(max_attributes, param_len)
        unique_gates = sorted(list(unique_gates))
        num_gate_types = len(unique_gates)
        gate_mapping = {gate: i for i, gate in enumerate(unique_gates)}
        new_in_channels = num_gate_types + max_attributes
        self.cnn_gate_mapping = gate_mapping
        self.cnn_max_attributes = max_attributes

        def cell_to_features(cell):
            gate = cell[0]
            params = cell[1:]
            if len(params) < max_attributes:
                params = params + [0] * (max_attributes - len(params))
            else:
                params = params[:max_attributes]
            one_hot = [0] * num_gate_types
            if gate in gate_mapping:
                one_hot[gate_mapping[gate]] = 1
            return one_hot + params

        cnn_data_list = []
        for idx, row in self.dataset.iterrows():
            conv_obj = row.get("ConvertedDataset", None)
            if conv_obj is None:
                raise ValueError("Missing 'ConvertedDataset' column in the row.")
            if isinstance(conv_obj, str):
                conv_dict = ast.literal_eval(conv_obj)
            else:
                conv_dict = conv_obj

            data_matrix = conv_dict.get("data", None)
            dim_dict = conv_dict.get("dimension", None)
            if data_matrix is None or dim_dict is None:
                raise ValueError("The ConvertedDataset dictionary must contain 'data' and 'dimension' keys.")
            n_total = int(dim_dict["n_total"])
            depth = int(dim_dict["depth"])
            in_channels = new_in_channels

            feature_matrix = [row_entry[1:] for row_entry in data_matrix]
            new_feature_matrix = []
            for row_entry in feature_matrix:
                new_row = []
                for cell in row_entry:
                    new_row.append(cell_to_features(cell))
                new_feature_matrix.append(new_row)
            sample_tensor = torch.tensor(new_feature_matrix, dtype=torch.float)
            sample_tensor = sample_tensor.permute(2, 0, 1)

            noisy = row.get("NoisyExpectation", None)[0]  # Overall noisy expectation (a single number)
            ideal = row.get("IdealExpectation", None)[0]  # Overall ideal expectation (a single number)
            zne = row.get("ZNEResults", None)  # Overall ZNE result (if available)
            if isinstance(zne, str):
                try:
                    zne = ast.literal_eval(zne)
                except Exception:
                    pass

            cnn_data_list.append({
                "cnn": sample_tensor,
                "in_channels": in_channels,
                "n_total": n_total,
                "depth": depth,
                "ideal": ideal,
                "noisy": noisy,
                "zne": zne
            })
        self.data_list = cnn_data_list

    def convert_graphs(self):
        """
        Converts dataset rows to PyTorch Geometric Data objects following a graph representation.

        Here, expectation values are overall (a single value per circuit) and stored as scalars.
        Each Data object contains:
          - x: Node feature tensor.
          - edge_index: Tensor of edges.
          - mlp_features: Additional features such as circuit depth.
          - y: Target overall ideal expectation value.
          - Additional attributes for overall noisy expectation and ZNE results.
        """
        pauli_one_hot_map = {'X': [1, 0, 0], 'Y': [0, 1, 0], 'Z': [0, 0, 1]}
        for _, row in self.dataset.iterrows():
            try:
                graph = row['ConvertedDataset']
                if isinstance(graph, str):
                    graph = ast.literal_eval(graph)
                nodes = graph.get('nodes', [])
                node_features = []
                for node in nodes:
                    vec = node.get('vector', [])
                    if len(vec) < 3:
                        raise ValueError("Vector does not contain enough elements")
                    gate_name, master_flag, slave_flag = vec[:3]
                    params = vec[3:]
                    one_hot = [0] * self.num_gate_types
                    if gate_name in self.mapping:
                        one_hot[self.mapping[gate_name]] = 1
                    final_vec = one_hot + [master_flag, slave_flag] + params
                    node_features.append(final_vec)
                x = torch.tensor(node_features, dtype=torch.float)

                edges = graph.get('edges', [])
                edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                              if edges else torch.empty((2, 0), dtype=torch.long))
                

                # Target overall ideal and Noisy expectation value.
                ideal_exp = torch.tensor(row['IdealExpectation'][0], dtype=torch.float)
                noisy_exp = row['NoisyExpectation'][0]
                
                # Here, instead of per-qubit, we assume an overall expectation as a scalar.
                mlp_features = torch.tensor([row['Depth'], noisy_exp], dtype=torch.float)

                data_obj = Data(x=x, edge_index=edge_index, y=ideal_exp, mlp_features=mlp_features)
                data_obj.noisy = torch.tensor(noisy_exp, dtype=torch.float)
                data_obj.ideal = ideal_exp

                zne_results = row.get('ZNEResults', None)
                if zne_results:
                    if isinstance(zne_results, str):
                        zne_results = ast.literal_eval(zne_results)
                    # Here we expect zne_results to be a single overall value per method.
                    for key, value in zne_results.items():
                        if value is not None:
                            setattr(data_obj, key, torch.tensor(value, dtype=torch.float))
                self.data_list.append(data_obj)
            except Exception as e:
                print(f"Error processing row with circuit_id {row.get('circuit_id', 'N/A')}: {e}")

    def convert_freq(self):
        """
        Converts dataset rows to a frequency representation DataFrame.

        For each sample, it creates columns for:
          - All keys from the frequency dictionary in 'ConvertedDataset'.
          - Circuit depth.
          - Overall noisy and ideal expectation values.
          - Overall ZNE results.
        
        Returns a dictionary representation.
        """
        rows_list = []
        for _, row in self.dataset.iterrows():
            try:
                row_dict = {}
                freq = row['ConvertedDataset']
                if isinstance(freq, str):
                    freq = ast.literal_eval(freq)
                row_dict.update(freq)
                row_dict['Depth'] = row['Depth']
                # Expect overall expectation values (not per-qubit arrays).
                row_dict['ideal'] = row['IdealExpectation'][0]
                row_dict['noisy'] = row['NoisyExpectation'][0]
                zne = row.get('ZNEResults', None)
                if zne is not None:
                    if isinstance(zne, str):
                        zne = ast.literal_eval(zne)
                    for key, value in zne.items():
                        row_dict[f'ZNE_{key}'] = value
                rows_list.append(row_dict)
            except Exception as e:
                print(f"Error processing row with circuit_id {row.get('circuit_id', 'N/A')}: {e}")
        try:
            df = pd.DataFrame(rows_list)
            df.fillna(0, inplace=True)
            self.data_list = df.to_dict(orient='list')
        except Exception as e:
            print(f"Error creating DataFrame: {e}")

    def get_data(self):
        """
        Returns the processed data.

        For data_type "cnn", each sample is returned as a dictionary with:
            'cnn': a flattened list representing the CNN feature tensor,
            'in_channels': the number of input channels,
            'n_total': total number of rows,
            'depth': number of time slices,
            'ideal': overall ideal expectation,
            'noisy': overall noisy expectation,
            and additional ZNE columns.
        
        For data_type "graph" and "freq", returns the corresponding data.
        """
        import ast
        if self.data_type == "cnn":
            combined_data = []
            for sample in self.data_list:
                cnn_tensor = sample["cnn"]
                cnn_flat = cnn_tensor.flatten().tolist()
                in_channels = sample["in_channels"]
                n_total = sample["n_total"]
                depth = sample["depth"]
                noisy = sample.get("noisy", None)
                ideal = sample.get("ideal", None)
                zne = sample.get("zne", None)
                if isinstance(zne, str):
                    try:
                        zne = ast.literal_eval(zne)
                    except Exception:
                        pass
                zne_columns = {}
                if isinstance(zne, dict):
                    for key, value in zne.items():
                        zne_columns[key] = value
                else:
                    zne_columns["zne"] = zne
                combined_sample = {
                    "cnn": cnn_flat,
                    "in_channels": in_channels,
                    "n_total": n_total,
                    "depth": depth,
                    "ideal": ideal,
                    "noisy": noisy,
                }
                combined_sample.update(zne_columns)
                combined_data.append(combined_sample)
            return combined_data
        elif self.data_type == "graph":
            return self.data_list
        elif self.data_type == "freq":
            try:
                return pd.DataFrame(self.data_list)
            except Exception as e:
                print(f"Error converting data list to DataFrame: {e}")
                return None
        else:
            print(f"Unsupported data type: {self.data_type}")
            return None

    def export_data(self, output_file, output_format="csv"):
        """
        Exports processed data to a file in CSV, JSON, or Pickle format.
        """
        try:
            data = self.get_data()
            if data is None:
                raise ValueError("No data available")
            if self.data_type == "cnn":
                df = pd.DataFrame(data)
            elif self.data_type == "graph":
                rows = []
                for data_obj in self.data_list:
                    row_dict = {}
                    for key in data_obj.keys():
                        value = data_obj[key]
                        row_dict[key] = json.dumps(value.tolist()) if hasattr(value, "tolist") else value
                    rows.append(row_dict)
                df = pd.DataFrame(rows)
            elif self.data_type == "freq":
                df = pd.DataFrame(self.data_list)
            else:
                raise ValueError(f"Unsupported data type: {self.data_type}")
            
            output_format = output_format.lower()
            if output_format == "csv":
                df.to_csv(output_file + ".csv", index=False, quoting=csv.QUOTE_ALL)
                print(f"Exported processed data to {output_file}.csv")
            elif output_format == "json":
                df.to_json(output_file + ".json", orient="records", lines=True)
                print(f"Exported processed data to {output_file}.json")
            elif output_format == "pickle":
                df.to_pickle(output_file + ".pkl")
                print(f"Exported processed data to {output_file}.pkl")
            else:
                raise ValueError(f"Output format '{output_format}' is not supported. Use 'csv', 'json', or 'pickle'.")
        except Exception as e:
            print(f"Error exporting data: {e}")

    def process(self):
        """
        Processes the dataset based on the specified data_type.
        """
        try:
            if self.data_type == "graph":
                self.convert_graphs()
            elif self.data_type == "freq":
                self.convert_freq()
            elif self.data_type == "cnn":
                self.convert_cnn()
            else:
                raise ValueError(f"Data type '{self.data_type}' is not implemented.")
        except Exception as e:
            print(f"Error processing dataset: {e}")
