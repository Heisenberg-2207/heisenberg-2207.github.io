import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Deep Learning imports for transformers
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# PyTorch Geometric imports
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

# Classic ML models
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# -----------------------
# Deep Learning Models
# -----------------------

# Simple GNN module that works on node features.
class SimpleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        in_channels: Dimension of node features.
        hidden_channels: Hidden dimension for GCN layers.
        out_channels: Number of outputs.
        """
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # Global mean pooling over all nodes in each graph.
        x = global_mean_pool(x, batch)
        return self.lin(x)


# Simple MLP for processing structured (per-qubit) features.
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        input_dim: Dimension of input features.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Simple CNN module for image classification or feature extraction.
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)  # assume input image size 28x28
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Simple Transformer module for sequence tasks.
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, num_layers, num_classes, dropout=0.1):
        """
        input_dim: Dimension of input features.
        nhead: Number of heads in the multihead attention model.
        hidden_dim: Dimension of the feedforward network model in nn.TransformerEncoderLayer.
        num_layers: Number of nn.TransformerEncoderLayer layers.
        num_classes: Number of output classes.
        dropout: Dropout value.
        """
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, src):
        """
        src: Tensor of shape (sequence_length, batch_size, input_dim)
        """
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        # Use the first token's state for classification
        output = self.fc_out(output[0])
        return output

# -----------------------
# Classical ML Models
# -----------------------

# Random Forest Regressor
class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, **kwargs):
        """
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of the tree.
        random_state: Seed for reproducibility.
        kwargs: Additional parameters for RandomForestRegressor.
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)


# XGBoost Regressor
class XGBoostModel:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, **kwargs):
        """
        n_estimators: Number of boosted trees.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        random_state: Seed for reproducibility.
        kwargs: Additional parameters for XGBRegressor.
        """
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)


# LightGBM Regressor
class LightGBMModel:
    def __init__(self, n_estimators=100, max_depth=-1, learning_rate=0.1, random_state=42, **kwargs):
        """
        n_estimators: Number of boosting iterations.
        max_depth: Maximum tree depth for base learners.
        learning_rate: Boosting learning rate.
        random_state: Seed for reproducibility.
        kwargs: Additional parameters for LGBMRegressor.
        """
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)


# CatBoost Regressor
class CatBoostModel:
    def __init__(self, iterations=1000, depth=6, learning_rate=0.1, random_state=42, verbose=False, **kwargs):
        """
        iterations: Number of boosting iterations.
        depth: Depth of the tree.
        learning_rate: Learning rate.
        random_state: Seed for reproducibility.
        verbose: Verbosity mode.
        kwargs: Additional parameters for CatBoostRegressor.
        """
        self.model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=verbose,
            **kwargs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)


# Support Vector Machine Regressor
class SVMModel:
    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', **kwargs):
        """
        C: Regularization parameter.
        epsilon: Epsilon in the epsilon-SVR model.
        kernel: Specifies the kernel type.
        kwargs: Additional parameters for SVR.
        """
        self.model = SVR(C=C, epsilon=epsilon, kernel=kernel, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)


# Linear Regression Model
class LinearRegressionModel:
    def __init__(self, fit_intercept=True, normalize=False, **kwargs):
        """
        fit_intercept: Whether to calculate the intercept for this model.
        normalize: This parameter is ignored when fit_intercept is set to False.
        kwargs: Additional parameters for LinearRegression.
        """
        self.model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)