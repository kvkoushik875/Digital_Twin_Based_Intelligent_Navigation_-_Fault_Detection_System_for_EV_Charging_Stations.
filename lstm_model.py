import torch
import torch.nn as nn

class FaultLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM expects input: (batch, seq_len, input_size) with batch_first=True
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Dropout before final classification
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer maps last hidden state → num_classes
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)

        # Use last layer’s hidden state for classification
        # h_n: (num_layers * num_directions, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size) or (batch, hidden_size * directions)

        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)  # (batch, num_classes)

        return logits
