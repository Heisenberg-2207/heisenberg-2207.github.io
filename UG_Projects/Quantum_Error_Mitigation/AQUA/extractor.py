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
      - 'NoisyExpectation': Noisy expectation values per qubit.
      - 'IdealExpectation': Ideal expectation values per qubit.
      - 'ObservablePattern': A string representing the Pauli observable per qubit (e.g., "XYZ").
      - 'ZNEResults': A string representing a dictionary with error mitigation results.
    """

    def __init__(self, filepath, n_qubits, backend, data_type="graph"):
        """
        Initializes the QuantumGraphExtractor.

        Parameters:
            dataset (pd.DataFrame): DataFrame containing quantum circuit data.
            n_qubits (int): Number of qubits in the circuits.
            backend: An object with an attribute `operation_names` (e.g., a quantum simulator backend).
            data_type (str): Type of conversion. Supported: "graph" and "freq" and "cnn.
        """
        self.dataset = self.load_dataset(filepath=filepath)
        self.n_qubits = n_qubits
        self.data_list = []  # Will hold processed objects (PyG Data objects or dicts).
        self.data_type = data_type.lower()

        try:
            # Retrieve gate operation names from the backend and extend with additional types.(only for graph)
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
        Convert the DataFrame (assumed to contain CNN data) to a list of dictionaries.
        
        The CSV is expected to contain at least the following columns:
        - "ConvertedDataset": a string representation of a dictionary with two keys:
                "data": a 2D list representing the CNN matrix with shape (n_total, depth+1),
                        where the first column contains row labels (e.g. 'q0', 'q1', ..., 'c0', 'c1', ...)
                        and the remaining columns contain flat lists representing cells in the form:
                            [gate_name, param1, param2, ..., paramN]
        - "dimension": a dictionary (or its string representation) with keys:
                "n_total": number of rows (qubits + classical bits),
                "depth": number of time slices (columns excluding the label column).
        - "NoisyExpectation": Noisy expectation values.
        - "IdealExpectation": Ideal expectation values.
        - "ZNEResults": ZNE expectation values (optional).
        
        In this updated version, the function computes a global one-hot mapping and maximum number 
        of parameters (max_attributes) across the dataset. For each cell, a feature vector is built as:
            one_hot(gate_name) + (padded/truncated parameters)
        so that the new in_channel dimension is:
            in_channels = num_gate_types + max_attributes.
        
        Each sample is then converted into a tensor of shape:
            (in_channels, n_total, depth)
        and stored along with its dimension metadata.
        
        Returns:
            list: A list where each element is a dictionary:
                { "cnn": cnn_tensor, "in_channels": in_channels, "n_total": n_total, "depth": depth,
                "noisy": noisy, "ideal": ideal, "zne": zne }
                with cnn_tensor being a PyTorch tensor.
        """

        # First, compute a global one-hot mapping and maximum parameter length.
        unique_gates = set()
        max_attributes = 0
        for _, row in self.dataset.iterrows():
            conv_str = row.get("ConvertedDataset", None)
            if conv_str is None:
                raise ValueError("Missing 'ConvertedDataset' column in a row.")
            if isinstance(conv_str, str):
                conv_dict = ast.literal_eval(conv_str)
            else:
                conv_dict = conv_str
            data_matrix = conv_dict.get("data", None)
            if data_matrix is None:
                raise ValueError("The ConvertedDataset dictionary must contain a 'data' key.")
            # Each row in data_matrix: first element is a label; subsequent cells are lists.
            for row_entry in data_matrix:
                for cell in row_entry[1:]:
                    if isinstance(cell, list) and len(cell) >= 1:
                        unique_gates.add(cell[0])
                        param_len = len(cell) - 1
                        if param_len > max_attributes:
                            max_attributes = param_len
        unique_gates = sorted(list(unique_gates))
        num_gate_types = len(unique_gates)
        gate_mapping = {gate: i for i, gate in enumerate(unique_gates)}
        # New in_channel dimension.
        new_in_channels = num_gate_types + max_attributes
        # Optionally store mapping info.
        self.cnn_gate_mapping = gate_mapping
        self.cnn_max_attributes = max_attributes

        # Define helper to convert a cell to a feature vector.
        # Each cell is of the form: [gate_name, param1, ..., paramN].
        # We produce: one_hot_vector (length=num_gate_types) concatenated with parameters padded/truncated to max_attributes.
        def cell_to_features(cell):
            gate = cell[0]
            params = cell[1:]
            # Pad or truncate parameters.
            if len(params) < max_attributes:
                params = params + [0] * (max_attributes - len(params))
            else:
                params = params[:max_attributes]
            one_hot = [0] * num_gate_types
            if gate in gate_mapping:
                one_hot[gate_mapping[gate]] = 1
            return one_hot + params  # length = num_gate_types + max_attributes

        cnn_data_list = []
        # Process each row in the dataset.
        for idx, row in self.dataset.iterrows():
            conv_str = row.get("ConvertedDataset", None)
            if conv_str is None:
                raise ValueError("Missing 'ConvertedDataset' column in the row.")
            if isinstance(conv_str, str):
                conv_dict = ast.literal_eval(conv_str)
            else:
                conv_dict = conv_str

            # Extract data matrix and dimension info.
            data_matrix = conv_dict.get("data", None)
            dim_dict = conv_dict.get("dimension", None)
            if data_matrix is None or dim_dict is None:
                raise ValueError("The ConvertedDataset dictionary must contain 'data' and 'dimension' keys.")

            n_total = int(dim_dict["n_total"])
            depth = int(dim_dict["depth"])
            # We no longer use the saved "in_channels" value.
            in_channels = new_in_channels

            # Remove the label column from each row.
            feature_matrix = [row_entry[1:] for row_entry in data_matrix]
            # Now feature_matrix should have shape (n_total, depth)
            # Each element (cell) is a flat list that we now convert to a feature vector.
            new_feature_matrix = []
            for row_entry in feature_matrix:
                new_row = []
                for cell in row_entry:
                    # Convert the cell to a feature vector.
                    new_row.append(cell_to_features(cell))
                new_feature_matrix.append(new_row)
            # new_feature_matrix has shape (n_total, depth, in_channels)
            sample_tensor = torch.tensor(new_feature_matrix, dtype=torch.float)
            # Permute to (in_channels, n_total, depth) as expected by CNNs.
            sample_tensor = sample_tensor.permute(2, 0, 1)

            # Extract additional expectation values.
            noisy = row.get("NoisyExpectation", None)
            ideal = row.get("IdealExpectation", None)
            zne = row.get("ZNEResults", None)
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
                "noisy": noisy,
                "ideal": ideal,
                "zne": zne
            })

        self.data_list = cnn_data_list


    def convert_graphs(self):
        """
        Converts dataset rows to PyTorch Geometric Data objects following a graph representation.

        Each Data object will contain:
          - x: Node feature tensor.
          - edge_index: Tensor of edges.
          - mlp_features: Additional features (noisy expectation, one-hot encoding of observable, depth).
          - y: Target ideal expectation values.
          - Additional attributes for noisy expectation and ZNE results.
        """
        pauli_one_hot_map = {'X': [1, 0, 0], 'Y': [0, 1, 0], 'Z': [0, 0, 1]}
        for _, row in self.dataset.iterrows():
            try:
                # Process the 'ConvertedDataset' column.
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

                # Process edges.
                edges = graph.get('edges', [])
                edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                              if edges else torch.empty((2, 0), dtype=torch.long))

                # Build additional MLP features.
                noisy_expvals = row['NoisyExpectation']
                pauli_str = row['ObservablePattern']
                mlp_features = [
                    [noisy_expvals[i]] + pauli_one_hot_map.get(pauli_str[i], [0, 0, 0]) + [row['Depth']]
                    for i in range(self.n_qubits)
                ]
                mlp_features = torch.tensor(mlp_features, dtype=torch.float)

                # Target ideal expectation values.
                ideal_expvals = torch.tensor(row['IdealExpectation'], dtype=torch.float)

                # Create the PyG Data object.
                data_obj = Data(x=x, edge_index=edge_index, y=ideal_expvals, mlp_features=mlp_features)
                data_obj.noisy = torch.tensor(noisy_expvals, dtype=torch.float)
                data_obj.ideal = ideal_expvals

                # Process ZNE results, if available.
                zne_results = row.get('ZNEResults', None)
                if zne_results:
                    if isinstance(zne_results, str):
                        zne_results = ast.literal_eval(zne_results)
                    for key, values in zne_results.items():
                        if values is not None:
                            setattr(data_obj, key, torch.tensor(values, dtype=torch.float))
                self.data_list.append(data_obj)
            except Exception as e:
                print(f"Error processing row with circuit_id {row.get('circuit_id', 'N/A')}: {e}")

    def convert_freq(self):
        """
        Converts dataset rows to a frequency representation DataFrame.

        For each sample, it creates columns for:
          - All keys from the frequency dictionary in 'ConvertedDataset'.
          - Pauli observables counts.
          - Circuit depth.
          - Noisy and ideal expectation values per qubit.
          - ZNE results per qubit.
          
        The result is stored in self.data_list.
        """
        rows_list = []
        for _, row in self.dataset.iterrows():
            try:
                row_dict = {}
                freq = row['ConvertedDataset']
                if isinstance(freq, str):
                    freq = ast.literal_eval(freq)
                row_dict.update(freq)

                # Process Pauli observables.
                pauli_str = row['ObservablePattern']
                row_dict['Pauli_Measure_X'] = pauli_str.count('X')
                row_dict['Pauli_Measure_Y'] = pauli_str.count('Y')
                row_dict['Pauli_Measure_Z'] = pauli_str.count('Z')

                row_dict['Depth'] = row['Depth']

                # Add noisy expectation values per qubit.
                noisy_exp = row['NoisyExpectation']
                for i, val in enumerate(noisy_exp):
                    row_dict[f'noisy_exp_q{i}'] = val

                # Add ideal expectation values per qubit.
                ideal_exp = row['IdealExpectation']
                for i, val in enumerate(ideal_exp):
                    row_dict[f'ideal_exp_q{i}'] = val

                # Process ZNE results.
                zne_results = row.get('ZNEResults', None)
                if zne_results is not None:
                    if isinstance(zne_results, str):
                        zne_results = ast.literal_eval(zne_results)
                    for key, values in zne_results.items():
                        for i, val in enumerate(values):
                            row_dict[f'ZNE_{key}_q{i}'] = val
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
        - 'cnn': a flattened list representing the CNN feature tensor,
                of shape (in_channels * n_total * depth)
        - 'in_channels': the number of input channels,
        - 'n_total': total number of rows (qubits + classical bits),
        - 'depth': the number of time slices,
        - 'ideal': Ideal expectation values,
        - 'noisy': Noisy expectation values,
        - plus separate columns for each key in the ZNE dictionary (if available).
        
        For data_type "graph" and "freq", returns the corresponding data.
        """
        import ast
        if self.data_type == "cnn":
            combined_data = []
            # self.data_list is a list of dictionaries (from convert_cnn), not a DataFrame.
            for sample in self.data_list:
                cnn_tensor = sample["cnn"]  # shape: (in_channels, n_total, depth)
                # Flatten the tensor to a 1D list.
                cnn_flat = cnn_tensor.flatten().tolist()
                in_channels = sample["in_channels"]
                n_total = sample["n_total"]
                depth = sample["depth"]
                # Extract expectation values stored in the sample dictionary.
                noisy = sample.get("noisy", None)
                ideal = sample.get("ideal", None)
                zne = sample.get("zne", None)
                # If zne is a string, try to parse it.
                if isinstance(zne, str):
                    try:
                        zne = ast.literal_eval(zne)
                    except Exception:
                        pass
                # If zne is a dictionary, expand its key/value pairs as separate columns.
                zne_columns = {}
                if isinstance(zne, dict):
                    for key, value in zne.items():
                        zne_columns[key] = value
                else:
                    zne_columns["zne"] = zne

                # Build the final output dictionary.
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
        
        For data_type "cnn", each exported row will contain:
        - cnn: Flattened CNN feature vector (list)
        - noisy: Noisy expectation values
        - ideal: Ideal expectation values
        - zne: ZNE expectation values
        """
        try:
            data = self.get_data()
            if data is None:
                raise ValueError("No data available")
            if self.data_type == "cnn":
                df = pd.DataFrame(data)
            elif self.data_type == "graph":
                # For graph type, convert tensor data to list for JSON serialization.
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
                # When saving CSV, each cell will be written as a string.
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


if __name__ == "__main__":

    backend = GenericBackendV2(5)  # Replace with actual backend if available.
    extractor = QuantumGraphExtractor(filepath='../../data/raw_data_cnn.csv', n_qubits=5, backend=backend, data_type="cnn")
    extractor.process()
    data_objects = extractor.get_data()
    if data_objects is not None:
        print("Data processing complete.")
    extractor.export_data("../../data/processed_data_cnn", output_format="csv")
