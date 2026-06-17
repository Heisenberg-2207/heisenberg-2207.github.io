import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch_geometric.data import Data

class Evaluator:
    def __init__(self, predicted, target, test_dataset):
        """
        predicted: numpy array of shape [n_samples, 1] (model predictions)
        target: numpy array of shape [n_samples, 1] (ideal expectation values)
        test_dataset: list of PyG Data objects, each containing attributes:
                      - noisy: unmitigated expectation (scalar)
                      - linear_zne: linear error mitigation result (scalar)
                      - richardson_zne: richardson error mitigation result (scalar)
                      - ideal: ideal expectation value (scalar)
        """
        self.predicted = predicted.reshape(-1, 1)
        self.target = target.reshape(-1, 1)
        self.test_dataset = test_dataset

        # Gather overall expectations from each sample in test_dataset.
        self.noisy = np.array([self._to_numpy(data.noisy).reshape(-1, 1) for data in test_dataset])
        self.linear = np.array([self._to_numpy(data.linear_zne).reshape(-1, 1) for data in test_dataset])
        self.richardson = np.array([self._to_numpy(data.richardson_zne).reshape(-1, 1) for data in test_dataset])
        self.ideal = np.array([self._to_numpy(data.ideal).reshape(-1, 1) for data in test_dataset])
        
        # Methods dictionary.
        self.methods = {
            'Noisy': self.noisy,
            'Linear': self.linear,
            'Richardson': self.richardson,
            'Predicted': self.predicted
        }
        
        # Number of samples.
        self.n_samples = self.ideal.shape[0]

    def _to_numpy(self, value):
        """Convert value (list, tensor, etc.) to a numpy array."""
        if isinstance(value, np.ndarray):
            return value
        elif isinstance(value, list):
            return np.array(value)
        elif torch.is_tensor(value):
            return value.cpu().detach().numpy()
        else:
            return np.array(value)

    def compute_metrics(self):
        """
        Compute error metrics (MSE, RMSE, MAE) for each method against ideal.
        Returns a dictionary mapping method names to a dict with keys 'MSE', 'RMSE', and 'MAE'.
        """
        metrics = {}
        ideal_flat = self.ideal.flatten()
        for method_name, values in self.methods.items():
            method_flat = values.flatten()
            mse_val = mean_squared_error(ideal_flat, method_flat)
            rmse_val = np.sqrt(mse_val)
            mae_val = mean_absolute_error(ideal_flat, method_flat)
            metrics[method_name] = {'MSE': mse_val, 'RMSE': rmse_val, 'MAE': mae_val}
        return metrics

    def print_metrics(self):
        """Print computed error metrics for each method."""
        metrics = self.compute_metrics()
        print("Error Metrics:")
        for method, scores in metrics.items():
            print(f"Method: {method}")
            for metric_name, value in scores.items():
                print(f"  {metric_name}: {value:.6f}")
            print()

    def plot_error_metrics(self, error_metric='MAE'):
        """
        Plot side-by-side subplots: error metric comparison and predictions vs ideal values.
        """
        metrics = self.compute_metrics()
        methods = list(metrics.keys())
        metric_vals = [metrics[m][error_metric.upper()] for m in methods]

        # Create subplots with 1 row, 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # First subplot: Error metric comparison
        ax1.bar(methods, metric_vals, color=['blue', 'green', 'orange', 'red'], alpha=0.8)
        ax1.set_xlabel("Method")
        ax1.set_ylabel(error_metric.upper())
        ax1.set_title(f"{error_metric.upper()} Comparison")
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Second subplot: Predictions vs Ideal
        x_axis = np.arange(self.n_samples)
        for method_name, values in self.methods.items():
            ax2.scatter(x_axis, values.flatten(), label=method_name, alpha=0.7)
        ax2.scatter(x_axis, self.ideal.flatten(), label='Ideal', color='black', marker='x')
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Expectation")
        ax2.set_title("Predicted vs True Values")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()  # Adjust spacing between subplots
        plt.show()


# Example usage:
if __name__ == "__main__":
    n_samples = 10
    # Generate dummy data: each value is a scalar.
    predicted = np.random.rand(n_samples, 1)
    target = np.random.rand(n_samples, 1)
    
    # Create a dummy test dataset with PyG Data objects.
    dummy_test_dataset = []
    for i in range(n_samples):
        data = Data()
        data.noisy = np.random.rand()         # overall noisy expectation (scalar)
        data.linear_zne = np.random.rand()      # overall linear ZNE (scalar)
        data.richardson_zne = np.random.rand()  # overall Richardson ZNE (scalar)
        data.ideal = np.random.rand()           # overall ideal expectation (scalar)
        dummy_test_dataset.append(data)
    
    evaluator = Evaluator(predicted, target, dummy_test_dataset)
    evaluator.print_metrics()
    
    # Plot box plots for error metrics for each method (combined over samples).
    # # Here, we only have one scalar per sample, so box plots will show distribution of errors.
    evaluator.plot_error_metrics(error_metric='MAE')
    # # Plot predictions vs. ideal values.
    evaluator.plot_results()
