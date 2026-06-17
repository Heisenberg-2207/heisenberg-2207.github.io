import ast
import csv
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

class DataPreProcessor:
    def __init__(self, filename, data_type="graph", n_qubits=None):
        """
        Parameters:
            filename (str): Path to the processed data file.
            data_type (str): "graph" to convert to PyTorch Geometric Data objects,
                             "freq" to extract features (X) and targets (Y),
                             "cnn" to extract CNN features and targets.
            n_qubits (int): Number of qubits (required for "graph" conversion and CNN reshaping).
        """
        self.filename = filename
        self.data_type = data_type.lower()
        self.n_qubits = n_qubits
        self.df = None
        self.converted_data = None


    def load_data(self):
        """Load the data file by inferring the file type from the file extension."""
        file_path = Path(self.filename)
        file_extension = file_path.suffix.lower()  # gets the extension, e.g. '.csv'
        
        try:
            if file_extension == '.csv':
                # Using robust quoting for CSV
                self.df = pd.read_csv(self.filename, quoting=csv.QUOTE_ALL)
            elif file_extension == '.json':
                self.df = pd.read_json(self.filename, orient="records", lines=True)
            elif file_extension in ['.pkl', '.pickle']:
                self.df = pd.read_pickle(self.filename)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
        except Exception as e:
            raise RuntimeError(f"Error loading data from {self.filename}: {e}")


    def convert_data(self):
        """
        Convert the loaded DataFrame into the desired format.
        
        Returns:
            If data_type == "graph": a list of PyTorch Geometric Data objects.
            If data_type == "freq": a tuple (X, Y) where X is the feature matrix and Y is the target matrix.
            If data_type == "cnn": a list of dictionaries, each containing:
                     'cnn': a reshaped CNN tensor (of shape in_channel x rows x depth),
                     'noisy': noisy expectation values,
                     'ideal': ideal expectation values,
                     'zne': ZNE expectation values (if available).
        """
        if self.df is None:
            self.load_data()
        if self.data_type == "graph":
            self.converted_data = self._convert_to_graph()
        elif self.data_type == "freq":
            self.converted_data = self._convert_to_freq()
        elif self.data_type == "cnn":
            self.converted_data = self._convert_to_cnn()
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")
        return self.converted_data

    def _convert_to_graph(self):
        """
        Convert each row of the DataFrame to a PyTorch Geometric Data object.
        All columns in the DataFrame are directly mapped as attributes.
        String values are attempted to be converted back to their original formats.
        """
        data_list = []
        for idx, row in self.df.iterrows():
            try:
                # Convert each value: if it's a string, try to evaluate it.
                row_dict = {k: (ast.literal_eval(v) if isinstance(v, str) else v)
                            for k, v in row.to_dict().items()}
                data_obj = Data(**row_dict)
                data_list.append(data_obj)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        return data_list

    def _convert_to_freq(self):
        """
        Convert the DataFrame to feature (X) and target (Y) arrays.
        
        All columns whose names contain the word "ideal" (case insensitive) are used as targets.
        The remaining columns are considered features.
        
        Returns:
            tuple: (X, Y) as NumPy arrays.
        """
        target_cols = [col for col in self.df.columns if 'ideal' in col.lower()]
        if not target_cols:
            raise ValueError("No target columns found with 'ideal' in their name.")
        feature_cols = [col for col in self.df.columns if col not in target_cols]
        X = self.df[feature_cols]
        Y = self.df[target_cols]
        return X.values, Y.values

    def _convert_to_cnn(self):
        """
        Reconstruct CNN tensors from the flattened CNN data stored in the DataFrame.
        
        The DataFrame is expected to contain the following columns:
        - "cnn": a flattened list (or its string representation) of length 
                (in_channels * n_total * depth) representing the CNN tensor.
        - "in_channels": an integer representing the number of channels.
        - "n_total": an integer representing the total number of rows (qubits + classical bits).
        - "depth": an integer indicating the number of time steps (depth) for that circuit.
        - "noisy": Noisy expectation values.
        - "ideal": Ideal expectation values.
        - "zne": ZNE expectation values (optional); if a dictionary, its key/value pairs will be expanded.
        
        This function reconstructs the CNN tensor by:
        1. Parsing the "cnn" column (if it is a string, using ast.literal_eval).
        2. Verifying the length equals in_channels * n_total * depth.
        3. Converting the flattened list into a PyTorch tensor and reshaping it to
            (in_channels, n_total, depth).
        
        Returns:
            list: A list where each element is a dictionary:
                { "cnn": cnn_tensor, "in_channels": in_channels, "n_total": n_total, "depth": depth,
                "noisy": noisy, "ideal": ideal, ... (one key per ZNE method, if zne is a dict) }
                with cnn_tensor being a PyTorch tensor.
        """

        cnn_data_list = []
        for idx, row in self.df.iterrows():
            # Parse the flattened CNN data.
            cnn_val = row.get("cnn", None)
            if cnn_val is None:
                raise ValueError(f"Missing 'cnn' column in row {idx}.")
            if isinstance(cnn_val, str):
                cnn_val = ast.literal_eval(cnn_val)
            
            # Get dimension metadata.
            in_channels = row.get("in_channels", None)
            if in_channels is None:
                raise ValueError(f"Missing 'in_channels' column in row {idx}.")
            in_channels = int(in_channels)
            
            n_total = row.get("n_total", None)
            if n_total is None:
                raise ValueError(f"Missing 'n_total' column in row {idx}.")
            n_total = int(n_total)
            
            depth = row.get("depth", None)
            if depth is None:
                raise ValueError(f"Missing 'depth' column in row {idx}.")
            depth = int(depth)
            
            # Verify that the flattened list has the expected length.
            expected = in_channels * n_total * depth
            if len(cnn_val) != expected:
                raise ValueError(f"Row {idx}: Expected flattened cnn length {expected} but got {len(cnn_val)}.")
            
            # Convert flattened list to tensor and reshape.
            cnn_tensor = torch.tensor(cnn_val, dtype=torch.float).reshape(in_channels, n_total, depth)
            
            # Extract expectation values.
            noisy = row.get("noisy", None)
            ideal = row.get("ideal", None)
            zne = row.get("zne", None)
            # If zne is a string, try to parse it.
            if isinstance(zne, str):
                try:
                    zne = ast.literal_eval(zne)
                except Exception:
                    pass
            
            # If zne is a dictionary, we expand its key/value pairs.
            zne_columns = {}
            if isinstance(zne, dict):
                for key, value in zne.items():
                    zne_columns[key] = value
            else:
                zne_columns["zne"] = zne

            # Create output dictionary for this row.
            out_dict = {
                "cnn": cnn_tensor,
                "in_channels": in_channels,
                "n_total": n_total,
                "depth": depth,
                "noisy": noisy,
                "ideal": ideal
            }
            out_dict.update(zne_columns)
            cnn_data_list.append(out_dict)
            
        return cnn_data_list

    def get_data(self):
        """
        Returns:
            If data_type == "graph": a list of PyTorch Geometric Data objects.
            If data_type == "freq": a tuple (X, Y) as NumPy arrays.
            If data_type == "cnn": a list of dictionaries (see _convert_to_cnn()).
        """
        if self.converted_data is None:
            raise ValueError("Data has not been converted yet. Call convert_data() first.")
        return self.converted_data

    # Only for troubleshooting
    def export_data(self, output_file, output_format="csv"):
        """
        Exports processed data to a file in CSV, JSON, or Pickle format.

        For data_type "cnn", each exported row will contain:
        - "cnn": Flattened CNN feature vector (list) of shape 
                [in_channels * n_total * depth]
        - "in_channels": number of input channels
        - "n_total": total number of rows (qubits + classical bits)
        - "depth": number of time slices
        - "ideal": Ideal expectation values
        - "noisy": Noisy expectation values
        - Plus separate columns for each key in the ZNE dictionary if available.
        """
        try:
            data = self.get_data()
            if data is None:
                raise ValueError("No data available")
            # For CNN data, data is a list of dictionaries.
            if self.data_type == "cnn":
                df = pd.DataFrame(data)
            elif self.data_type == "graph":
                rows = []
                for data_obj in self.data_list:
                    try:
                        row_dict = {}
                        for key in data_obj.keys():
                            value = data_obj[key]
                            row_dict[key] = json.dumps(value.tolist()) if hasattr(value, "tolist") else value
                        rows.append(row_dict)
                    except Exception as e:
                        print(f"Error converting data object to dict: {e}")
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



if __name__ == "__main__":
    # Example usage:
    # Specify the file path, file format, and desired data type.
    # For CNN data, the CSV should have columns "cnn", "depth", "noisy", "ideal", and optionally "zne".
    filename = '../../data/processed_data_cnn.csv'
    data_type = "cnn"  # Change to "graph" or "freq" as needed.
    n_qubits = 5  # Must be provided for CNN conversion.
    
    processor = DataPreProcessor(filename, data_type=data_type, n_qubits=n_qubits)
    converted_data = processor.convert_data()
    
    if data_type == "graph":
        # Print the first 3 PyTorch Geometric Data objects.
        for data_obj in converted_data[:3]:
            print(data_obj)
    elif data_type == "cnn":
        # Print the first 2 CNN entries.
        for entry in converted_data[:2]:
            print("CNN tensor shape:", entry["cnn"].shape)
            print("Noisy:", entry["noisy"])
            print("Ideal:", entry["ideal"])
            print("ZNE:", entry["zne"])
    else:
        X, Y = converted_data
        print("Feature shape:", X.shape)
        print("Target shape:", Y.shape)
