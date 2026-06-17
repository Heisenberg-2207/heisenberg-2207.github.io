import numpy as np
import torch
from torch import nn
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

try:
    from QEM.base_models import *
except ImportError:
    from base_models import *

class QuantumGraphTrainer:
    def __init__(self, data_list, n_qubits, hidden_channels=32, mlp_hidden=16,
                 batch_size=4, lr=0.01, epochs=50, test_size=0.2,
                 mult_mlp=1, mult_graph=1):
        """
        data_list: List of PyTorch Geometric Data objects.
        n_qubits: Number of qubits per circuit.
        Multipliers:
          - mult_mlp: Scales the output from the MLP branch.
          - mult_graph: Scales the output from the GNN branch.
          
        In this version, the final prediction is a single scalar overall expectation.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_qubits = n_qubits
        self.mult_mlp = mult_mlp
        self.mult_graph = mult_graph

        # Convert attributes to torch tensors if necessary.
        for data in data_list:
            if isinstance(data.x, list):
                data.x = torch.tensor(data.x, dtype=torch.float)
            if isinstance(data.mlp_features, list):
                data.mlp_features = torch.tensor(data.mlp_features, dtype=torch.float)
            if isinstance(data.y, list):
                data.y = torch.tensor(data.y, dtype=torch.float)
            if isinstance(data.edge_index, list):
                data.edge_index = torch.tensor(data.edge_index, dtype=torch.long)

        # Split into train and test sets.
        train_data, test_data = train_test_split(data_list, test_size=test_size, random_state=42)
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Determine node feature dimension from the first data object.
        in_channels = data_list[0].x.size(1)
        # GNN branch: set output_dim=1 for overall expectation.
        self.gnn = SimpleGNN(in_channels, hidden_channels, out_channels=1)
        # MLP branch expects structured features of dimension 2: [Depth, Overall Noisy Expectation]
        self.mlp = MLP(input_dim=2, hidden_dim=mlp_hidden, output_dim=1)
        # Final MLP combines outputs from both branches (2 scalars) and outputs a single scalar.
        self.final_mlp = MLP(input_dim=2, hidden_dim=mlp_hidden, output_dim=1)

        # Optimizer and loss function.
        self.optimizer = torch.optim.Adam(
            list(self.gnn.parameters()) +
            list(self.mlp.parameters()) +
            list(self.final_mlp.parameters()),
            lr=lr
        )
        self.criterion = nn.MSELoss()

        self.train_losses = []  # For visualization

    def train(self):
        self.gnn.train()
        self.mlp.train()
        self.final_mlp.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()

                # Graph branch through GNN.
                gnn_out = self.gnn(batch) * self.mult_graph  # shape: [batch_size, 1]

                # MLP branch: here we assume batch.mlp_features is already a [batch_size, 2] tensor.
                mlp_out = self.mlp(batch.mlp_features.view(-1, 2)) * self.mult_mlp  # shape: [batch_size, 1]

                # Concatenate the two scalar outputs: shape becomes [batch_size, 2].
                combined_out = torch.cat([gnn_out, mlp_out], dim=1)
                final_pred = self.final_mlp(combined_out)  # shape: [batch_size, 1]

                # Target overall expectation is expected to be a scalar per sample.
                target_y = batch.y.view(batch.num_graphs, -1)  # shape: [batch_size, 1]
                loss = self.criterion(final_pred, target_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.num_graphs

            avg_loss = total_loss / len(self.train_loader.dataset)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch+1:02d}, Loss: {avg_loss:.4f}")

    def evaluate(self):
        self.gnn.eval()
        self.mlp.eval()
        self.final_mlp.eval()

        predictions, targets = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                mlp_feats = batch.mlp_features.view(-1, 2)
                gnn_out = self.gnn(batch) * self.mult_graph  # shape: [batch_size, 1]
                mlp_out = self.mlp(mlp_feats) * self.mult_mlp  # shape: [batch_size, 1]
                combined_out = torch.cat([gnn_out, mlp_out], dim=1)  # shape: [batch_size, 2]
                final_pred = self.final_mlp(combined_out)  # shape: [batch_size, 1]
                predictions.append(final_pred.cpu().numpy())
                targets.append(batch.y.view(batch.num_graphs, -1).cpu().numpy())
        return np.vstack(predictions), np.vstack(targets)

    def plot_results(self):
        # Plot training loss curve.
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, marker='o', label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot predictions vs. targets (overall expectation) for test samples.
        predictions, targets = self.evaluate()
        # For clarity, we use a scatter plot.
        plt.figure(figsize=(8, 5))
        plt.scatter(np.arange(len(targets)), targets, color='blue', label="True Overall Expectation", marker='o')
        plt.scatter(np.arange(len(predictions)), predictions, color='red', label="Predicted Overall Expectation", marker='x')
        plt.xlabel("Test Sample Index")
        plt.ylabel("Expectation")
        plt.title("Predicted vs True Overall Ideal Expectation")
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Assume data_list is a list of PyG Data objects loaded from your processed dataset.
    # Each Data object must have attributes:
    # - x: Node feature tensor.
    # - mlp_features: Tensor of shape [n_samples, 2] with [Depth, Overall Noisy Expectation].
    # - y: Target overall ideal expectation (scalar).
    # - edge_index: Edge index tensor.
    # For demonstration, we assume data_list is provided.
    #
    # For this example, we load data_list from a pickle or CSV using your own methods.
    # Here, we use an empty list as placeholder.
    data_list = []  # Replace with actual loading of PyG Data objects.
    
    # For demonstration purposes, if data_list is empty, we skip training.
    if len(data_list) == 0:
        print("No data available for training.")
    else:
        trainer = QuantumGraphTrainer(
            data_list=data_list,
            n_qubits=5,
            hidden_channels=32,
            mlp_hidden=16,
            batch_size=4,
            lr=0.01,
            epochs=50,
            test_size=0.2,
            mult_mlp=1,
            mult_graph=1
        )
        trainer.train()
        trainer.plot_results()
