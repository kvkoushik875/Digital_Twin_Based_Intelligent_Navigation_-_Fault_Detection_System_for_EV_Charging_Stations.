"""PyTorch autoencoder for EV charging anomaly detection."""

import torch
from torch import nn, Tensor


class EVAutoencoder(nn.Module):
    """
    Autoencoder that learns normal EV-charging sensor patterns.

    Architecture:
        Input(14) -> 64 -> 32 -> 16 -> Latent(8)
        Latent(8) -> 16 -> 32 -> 64 -> Output(14)
    """

    def __init__(
        self,
        input_dim: int = 14,
        latent_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero.")

        if latent_dim <= 0:
            raise ValueError("latent_dim must be greater than zero.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),

            nn.Linear(16, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, input_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize linear layers using Kaiming initialization."""

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight,
                    nonlinearity="relu",
                )

                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def encode(self, inputs: Tensor) -> Tensor:
        """Convert normalized sensor values into latent features."""

        self._validate_input(inputs)
        return self.encoder(inputs)

    def decode(self, latent: Tensor) -> Tensor:
        """Reconstruct sensor features from latent features."""

        return self.decoder(latent)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return reconstructed sensor values."""

        latent = self.encode(inputs)
        return self.decode(latent)

    def reconstruction_error(self, inputs: Tensor) -> Tensor:
        """
        Calculate one mean-squared reconstruction error per record.

        Returns:
            Tensor with shape: [batch_size]
        """

        reconstructed = self.forward(inputs)

        return torch.mean(
            torch.square(inputs - reconstructed),
            dim=1,
        )

    def _validate_input(self, inputs: Tensor) -> None:
        if inputs.ndim != 2:
            raise ValueError(
                "Input must have shape [batch_size, input_dim]."
            )

        if inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"received {inputs.shape[1]}."
            )


if __name__ == "__main__":
    model = EVAutoencoder(input_dim=14)

    sample_batch = torch.rand(
        size=(4, 14),
        dtype=torch.float32,
    )

    reconstructed_batch = model(sample_batch)
    anomaly_scores = model.reconstruction_error(sample_batch)

    print(model)
    print("Input shape:", sample_batch.shape)
    print("Output shape:", reconstructed_batch.shape)
    print("Anomaly scores:", anomaly_scores)