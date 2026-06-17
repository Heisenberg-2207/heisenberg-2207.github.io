import numpy as np
from qiskit.converters import circuit_to_dag

class CircuitConverter:
    """
    A class to convert a quantum circuit into different dataset representations.
    
    Currently implemented conversions:
      - "graph": Converts a circuit to a graph representation (nodes and edges).
      - "frequency": Converts a circuit to a frequency representation.
    
    Future conversion types can be added and selected via the `conversion_type` argument.
    """
    
    def __init__(self, max_gate_attributes=4, bin_min=None, bin_max=None, bin_size=None, mode='bin'):
        """
        Initialize the converter.
        
        Parameters:
            max_gate_attributes (int): Maximum number of gate parameters to include.
            bin_min (float): Minimum value for binning (for frequency conversion).
            bin_max (float): Maximum value for binning (for frequency conversion).
            bin_size (float): Bin size (for frequency conversion).
            mode (str): Mode for frequency conversion. Options:
                        'bin' - global binning,
                        'bin_gate' - separate bins for each gate,
                        'bin_gate_attr' - separate bins for each gate per attribute.
        """
        self.max_gate_attributes = max_gate_attributes
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_size = bin_size
        self.mode = mode

    def convert(self, circ, conversion_type="graph"):
        """
        Convert a given circuit using the selected conversion method.
        
        Parameters:
            circ (QuantumCircuit): The quantum circuit to convert.
            conversion_type (str): The type of conversion. Options: "graph" or "freq" or "cnn".
            idx (int): Starting index for node IDs (used in some conversions).
            
        Returns:
            tuple: The converted dataset (format depends on conversion_type)
                   and, if applicable, the updated global index.
        """
        if conversion_type.lower() == "graph":
            return self.convert_circuit_to_graph(circ)
        elif conversion_type.lower() == "freq":
            return self.convert_circuit_to_frequency(circ)
        elif conversion_type.lower() == "cnn":
            return self.convert_circuit_to_cnn(circ)
        else:
            raise ValueError(f"Conversion type '{conversion_type}' is not supported.")

    def convert_circuit_to_graph(self, circ):
        """
        Convert a given quantum circuit to a graph representation.
        
        The graph is represented as a dictionary with:
          - nodes: a list of node dictionaries, each with an "id" and "vector"
          - edges: a list of [src, tgt] pairs representing directed edges.
        
        Parameters:
            circ (QuantumCircuit): The circuit to convert.
        
        Returns:
            tuple: (graph)
                graph (dict): Dictionary with keys "nodes" and "edges".
        """
        idx = 0
        dag = circuit_to_dag(circ)
        nodes = []
        node_id_mapping = {}
        
        for node in dag.topological_nodes():
            # Determine node type based on its attributes or class name.
            if hasattr(node, "type"):
                ntype = node.type
            else:
                cls_name = node.__class__.__name__
                if cls_name == "DAGInNode":
                    ntype = "in"
                elif cls_name == "DAGOutNode":
                    ntype = "out"
                elif cls_name == "DAGOpNode":
                    ntype = "op"
                else:
                    ntype = "unknown"
            
            # For inputs and outputs, use the type as the gate name.
            if ntype in ["in", "out"]:
                gate_name = ntype
            elif ntype == "op":
                gate_name = node.op.name
            else:
                gate_name = "unknown"
            
            # Get parameters from the gate (if available)
            if ntype == "op" and hasattr(node.op, "params") and node.op.params:
                params = list(node.op.params)
            else:
                params = []
            
            # Pad or truncate parameters to match max_gate_attributes.
            if len(params) < self.max_gate_attributes:
                params.extend([0] * (self.max_gate_attributes - len(params)))
            else:
                params = params[:self.max_gate_attributes]
            
            # For multi-qubit operations, create a node for each operand.
            if ntype == "op" and len(node.qargs) > 1:
                copy_ids = []
                for pos in range(len(node.qargs)):
                    master_flag, slave_flag = (1, 0) if pos == 0 else (0, 1)
                    node_vector = [gate_name, master_flag, slave_flag] + params
                    node_dict = {"id": idx, "vector": node_vector}
                    nodes.append(node_dict)
                    copy_ids.append(idx)
                    idx += 1
                node_id_mapping[node] = copy_ids[0]
            else:
                node_vector = [gate_name, 0, 0] + params
                node_dict = {"id": idx, "vector": node_vector}
                nodes.append(node_dict)
                node_id_mapping[node] = idx
                idx += 1
        
        # Create edges based on the DAG connectivity.
        edges = []
        for edge in dag.edges():
            src, tgt, _ = edge
            src_id = node_id_mapping.get(src, None)
            tgt_id = node_id_mapping.get(tgt, None)
            if src_id is not None and tgt_id is not None:
                edges.append([src_id, tgt_id])
        
        graph = {"nodes": nodes, "edges": edges}
        return graph

    def convert_circuit_to_frequency(self, circ):
        """
        Convert a quantum circuit to a frequency representation.
        
        It discretizes the angle space from bin_min to bin_max with bin_size and then
        counts the frequency of gate parameters falling in each bin.
        
        Modes:
          - 'bin': Global binning across all gate parameters. Keys: "bin_lower_upper".
          - 'bin_gate': Separate bins for each gate. Keys: "bin_<gate>_lower_upper".
          - 'bin_gate_attr': Separate bins per gate per attribute index. Keys: "bin_<gate>_<attr>_lower_upper".
            If a gate has more parameters than max_gate_attributes, an error is raised.
        
        Multi-qubit gates are automatically detected: the first qubit is counted as the master,
        and the rest as slaves (keys: "<gate>_master", "<gate>_slaves").
        
        Parameters:
            circ (QuantumCircuit): The circuit to convert.

        Returns:
            tuple: (freq, idx)
                freq (dict): Frequency representation of the circuit.
        """
        # Ensure binning range is provided.
        if self.bin_min is None or self.bin_max is None:
            raise ValueError("bin_min and bin_max must be specified for frequency conversion.")

        # Pre-compute bins from bin_min to bin_max.
        bin_edges = []
        current = self.bin_min
        while current < self.bin_max:
            bin_edges.append((current, current + self.bin_size))
            current += self.bin_size

        # Initialize frequency dictionary based on mode.
        freq = {}
        if self.mode == 'bin':
            for lower, upper in bin_edges:
                key = f"bin_{lower:.2f}_{upper:.2f}"
                freq[key] = 0
        elif self.mode == 'bin_gate':
            # Get unique gate names.
            gate_names = set(instr.operation.name for instr in circ.data)
            for gate in gate_names:
                for lower, upper in bin_edges:
                    key = f"bin_{gate}_{lower:.2f}_{upper:.2f}"
                    freq[key] = 0
        elif self.mode == 'bin_gate_attr':
            gate_names = set(instr.operation.name for instr in circ.data)
            for gate in gate_names:
                for attr_idx in range(self.max_gate_attributes):
                    for lower, upper in bin_edges:
                        key = f"bin_{gate}_{attr_idx}_{lower:.2f}_{upper:.2f}"
                        freq[key] = 0
        else:
            raise ValueError("Invalid mode specified. Use 'bin', 'bin_gate' or 'bin_gate_attr'.")

        # Process each instruction.
        for instr in circ.data:
            gate = instr.operation
            qubits = instr.qubits
            # For multi-qubit gates, count master and slave frequencies.
            if len(qubits) > 1:
                master_key = f"{gate.name}_master"
                slaves_key = f"{gate.name}_slaves"
                if master_key not in freq:
                    freq[master_key] = 0
                if slaves_key not in freq:
                    freq[slaves_key] = 0
                freq[master_key] += 1
                freq[slaves_key] += (len(qubits) - 1)
            else:
                # For single-qubit gates, count overall gate frequency.
                key = gate.name
                if key not in freq:
                    freq[key] = 0
                freq[key] += 1

            # Process gate parameters for binning.
            if self.mode == 'bin_gate_attr' and len(gate.params) > self.max_gate_attributes:
                raise ValueError(f"Gate {gate.name} has {len(gate.params)} parameters which exceeds max_gate_attributes {self.max_gate_attributes}")
            for attr_idx, param in enumerate(gate.params):
                try:
                    param_val = float(param)
                except Exception:
                    continue
                if not (self.bin_min <= param_val < self.bin_max):
                    continue  # Skip parameters outside the defined range.
                # Determine bin index.
                bin_index = int((param_val - self.bin_min) // self.bin_size)
                lower, upper = bin_edges[bin_index]
                if self.mode == 'bin':
                    bin_key = f"bin_{lower:.2f}_{upper:.2f}"
                elif self.mode == 'bin_gate':
                    bin_key = f"bin_{gate.name}_{lower:.2f}_{upper:.2f}"
                elif self.mode == 'bin_gate_attr':
                    bin_key = f"bin_{gate.name}_{attr_idx}_{lower:.2f}_{upper:.2f}"
                freq[bin_key] = freq.get(bin_key, 0) + 1

        return freq

    def convert_circuit_to_cnn(self, circ):
        """
        Convert a quantum circuit to a CNN-readable 2D matrix with row labels.
        
        The output is a dictionary with:
        - "data": a 2D list (of shape (n_total, depth+1)) where the first column contains row labels 
                    (e.g. 'q0', 'q1', ..., 'c0', 'c1', ...) and every other cell is a flat list of length 
                    (1 + self.max_gate_attributes) representing an operation.
        - "dimension": a dict with keys:
                "n_total": number of rows (qubits + classical bits),
                "depth": number of time slices (columns excluding the label column).
        
        Each cell (except the first column) is a flat list of the form:
            [gate_name, param1, param2, ..., paramN]
        where N is self.max_gate_attributes.
        
        Special handling:
        - Barrier: all involved qubits get ["barrier", ...] (parameters not used).
        - Measure: the qubit row gets ["measure_master", ...] and the corresponding classical bit row gets
                    ["measure_slave", ...].
        - Reset: each qubit involved gets ["reset", ...].
        - For multi-qubit operations: the first qubit is updated as [<gate>_master, ...] and the remaining qubits
            as [<gate>_slave, ...].
        
        Cells not updated remain as the default "noop" placeholder.
        
        Parameters:
            circ (QuantumCircuit): The circuit to convert.
            
        Returns:
            dict: { "data": <CNN matrix as a list>, "dimension": {"n_total": ..., "depth": ...} }
        """
        # Create mappings for qubits and classical bits to rows.
        n_qubits = len(circ.qubits)
        n_clbits = len(circ.clbits)
        total_rows = n_qubits + n_clbits

        # Generate row labels.
        row_labels = [f"q{i}" for i in range(n_qubits)] + [f"c{i}" for i in range(n_clbits)]
        
        qubit_map = {q: i for i, q in enumerate(circ.qubits)}
        clbit_map = {c: i + n_qubits for i, c in enumerate(circ.clbits)}
        
        # Convert the circuit to a DAG.
        dag = circuit_to_dag(circ)
        
        # Assign a time step (layer index) to each node.
        level_dict = {}
        for node in dag.topological_nodes():
            cls_name = node.__class__.__name__
            if cls_name == "DAGInNode":
                node_type = "in"
            elif cls_name == "DAGOutNode":
                node_type = "out"
            elif cls_name == "DAGOpNode":
                node_type = "op"
            else:
                node_type = "unknown"
            
            if node_type == "in":
                level_dict[node] = 0
            else:
                pred_levels = [level_dict[p] for p in dag.predecessors(node) if p in level_dict]
                level_dict[node] = max(pred_levels) + 1 if pred_levels else 0

        # Determine depth (number of time steps).
        depth = max(level_dict.values()) + 1

        # Helper: produce a flat list from a gate name and parameter list.
        def flatten_cell(gate, params):
            return [gate] + params

        # Use self.max_gate_attributes to create a default cell.
        noop_flat = flatten_cell("noop", [0] * self.max_gate_attributes)

        # Initialize the CNN matrix with shape (total_rows, depth+1).
        # First column: row labels; other columns: default "noop".
        cnn_matrix = np.empty((total_rows, depth + 1), dtype=object)
        for i in range(total_rows):
            cnn_matrix[i, 0] = row_labels[i]
        for i in range(total_rows):
            for j in range(1, depth + 1):
                # Copy to avoid sharing the same list reference.
                cnn_matrix[i, j] = noop_flat.copy()

        # For each node (operation) in the DAG, update the corresponding cell.
        for node in dag.topological_nodes():
            cls_name = node.__class__.__name__
            # Skip non-operational nodes.
            if cls_name in ["DAGInNode", "DAGOutNode"]:
                continue
            
            # Determine the time slice (column index) for this node.
            t = level_dict[node] + 1  # +1 to account for label column

            # Extract gate parameters, pad or truncate to self.max_gate_attributes.
            params = list(node.op.params) if hasattr(node.op, "params") and node.op.params else []
            if len(params) < self.max_gate_attributes:
                params.extend([0] * (self.max_gate_attributes - len(params)))
            else:
                params = params[:self.max_gate_attributes]
            
            # Determine the gate name.
            gate_name = node.op.name if hasattr(node.op, "name") else "unknown"
            
            # Create the flat cell value based on the type.
            if gate_name == "barrier":
                cell_val = flatten_cell("barrier", [0] * self.max_gate_attributes)
                # For barrier, update all qubits in node.qargs.
                for qubit in node.qargs:
                    cnn_matrix[qubit_map[qubit], t] = cell_val
            elif gate_name == "measure":
                # Expecting one qubit and one classical bit.
                if node.qargs and node.cargs:
                    cell_val_q = flatten_cell("measure_master", [0] * self.max_gate_attributes)
                    cell_val_c = flatten_cell("measure_slave", [0] * self.max_gate_attributes)
                    cnn_matrix[qubit_map[node.qargs[0]], t] = cell_val_q
                    cnn_matrix[clbit_map[node.cargs[0]], t] = cell_val_c
            elif gate_name == "reset":
                cell_val = flatten_cell("reset", [0] * self.max_gate_attributes)
                for qubit in node.qargs:
                    cnn_matrix[qubit_map[qubit], t] = cell_val
            else:
                # For multi-qubit operations, assign master/slave labels.
                if len(node.qargs) > 1:
                    for pos, qubit in enumerate(node.qargs):
                        suffix = "master" if pos == 0 else "slave"
                        cell_val = flatten_cell(f"{gate_name}_{suffix}", params)
                        cnn_matrix[qubit_map[qubit], t] = cell_val
                else:
                    # Single-qubit operation.
                    if node.qargs:
                        cnn_matrix[qubit_map[node.qargs[0]], t] = flatten_cell(gate_name, params)
        
        # Convert the NumPy array to a Python list.
        cnn_list = np.array(cnn_matrix, dtype=object).tolist()
        dimensions = {"n_total": total_rows, "depth": depth}
        
        return {"data": cnn_list, "dimension": dimensions}



# Example usage:
if __name__ == "__main__":
    from qiskit.circuit.random import random_circuit
    # Generate a random circuit (without measurements)
    n_qubits = 3
    depth = 4
    circuit = random_circuit(n_qubits, depth, measure=False, seed=42)
    
    # Create a converter with settings for both graph and frequency conversions.
    converter = CircuitConverter(
        max_gate_attributes=3,
        bin_min=-3.14,
        bin_max=3.14,
        bin_size=0.5,
        mode='bin'  # Try 'bin_gate' or 'bin_gate_attr' as alternatives.
    )
    
    # Convert to graph representation.
    dataset_graph = converter.convert(circuit, conversion_type="graph")
    print("Converted dataset (graph representation):")
    print(dataset_graph)
    
    # Convert to frequency representation.
    dataset_freq = converter.convert(circuit, conversion_type="freq")
    print("\nConverted dataset (frequency representation):")
    print(dataset_freq)

    # Convert to CNN representation.
    dataset_cnn = converter.convert(circuit, conversion_type="cnn")
    print("\nConverted dataset (cnn representation):")
    print(dataset_cnn)
