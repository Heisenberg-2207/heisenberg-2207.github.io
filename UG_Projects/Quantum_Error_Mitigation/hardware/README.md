# Hardware Experiments — IBM Q Brisbane

Validates the simulator-trained QEM approach against real noise on **IBM Q Brisbane** (127-qubit superconducting hardware), via Qiskit Runtime.

## Files

| File | Description |
|---|---|
| `main.ipynb` | Primary hardware notebook: connects to `ibm_brisbane` via `QiskitRuntimeService`, pulls the device noise model, and runs the encoding/prediction pipeline against it. |
| `backup_main.ipynb` | Backup/earlier version of the main hardware notebook. |
| `data_gen.ipynb` | Generates random circuits, transpiles them for `ibm_brisbane`, and extracts the active physical qubits / hardware-aware virtual circuit mapping used for data generation. |
| `hardware_encoding.ipynb` | Encodes transpiled hardware circuits (with their actual physical-qubit layout) into the graph representation used by the GNN models. |
| `gnn_predictor.ipynb` | `GNNPredictor` — an edge-feature-aware GNN (`NNConv` + `GCNConv` + global mean pooling) trained to predict the ideal expectation value from a noisy hardware circuit graph, conditioned on the full 127-qubit topology (`qubit_dim=127`). |
| `model_study.ipynb` | Compares GCN- vs. **GAT**-based architectures (`GCNModelJoint`, using `GATConv`) on the hardware dataset — this is where the GAT results referenced in the top-level project README come from. |
| `ibm_brisbane_calibrations_2025-04-05T06_27_38Z.csv` | Device calibration snapshot (gate errors, readout errors, T1/T2) pulled from IBM Quantum at the time of data generation. |
| `gnn_dataset_20250409_013605.pt`, `gnn_dataset_20250409_225849.pt` | Pre-processed PyTorch Geometric datasets (two generation runs) of encoded hardware circuits, ready for training. |

## Notes

Connecting to IBM Quantum requires a valid `QiskitRuntimeService` account token saved locally via `QiskitRuntimeService.save_account(...)` — **do not hardcode tokens directly in notebooks**; the account-saving line in `main.ipynb`/`backup_main.ipynb` is intentionally commented out and redacted.
