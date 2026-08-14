import torch
import torch.nn as nn


class SensorAutoencoder(nn.Module):
    """Symmetric encoder-decoder MLP with a bottleneck.

    The bottleneck forces the network to learn a compressed representation
    of normal sensor behavior; reconstruction error on unseen (faulty)
    patterns is the anomaly signal.
    """

    def __init__(self, n_features, hidden_dims=(12, 8), bottleneck_dim=4, dropout=0.1):
        super().__init__()

        encoder_layers = []
        in_dim = n_features
        for h in hidden_dims:
            encoder_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = bottleneck_dim
        for h in reversed(hidden_dims):
            decoder_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, n_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)
