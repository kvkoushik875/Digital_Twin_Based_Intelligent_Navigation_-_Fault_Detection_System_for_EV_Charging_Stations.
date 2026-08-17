import torch
import torch.nn as nn


class MaintenanceAutoencoder(nn.Module):
    """Small encoder-decoder MLP, same unsupervised principle as
    fault_engine's SensorAutoencoder: learn to reconstruct what a
    station's health-engine profile normally looks like, so stations
    whose profile is hard to reconstruct (high error) stand out as
    needing attention. Scaled down from fault_engine's architecture
    since there are only a few health-engine features here, not 16
    sensor readings.
    """

    def __init__(self, n_features, hidden_dim=4, bottleneck_dim=2, dropout=0.05):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
