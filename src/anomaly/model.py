"""
MPED-RNN: Message-Passing Encoder-Decoder RNN for skeleton-based anomaly detection.

Architecture:
    - Encoder: LSTM that encodes a sequence of skeleton poses into a latent representation
    - Decoder (Reconstruction): LSTM that reconstructs the input sequence in reverse
    - Decoder (Prediction): LSTM that predicts future skeleton poses

Anomaly scoring:
    - Trained on normal skeleton sequences only (unsupervised)
    - Anomaly score = reconstruction error + prediction error (MSE)
    - High error indicates the sequence deviates from learned normal patterns

Reference: Morais et al., "Learning Regularity in Skeleton Trajectories for
Anomaly Detection in Videos" (CVPR 2019)
"""

import torch
import torch.nn as nn

from src.anomaly.data import INPUT_DIM


class MPEDRNN(nn.Module):
    """
    Encoder-Decoder LSTM with dual heads for reconstruction and prediction.

    The encoder processes the input skeleton sequence. Two separate decoders
    reconstruct the input (reversed) and predict future frames, respectively.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pred_len: int = 6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Reconstruction decoder
        self.decoder_recon = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_recon = nn.Linear(hidden_dim, input_dim)

        # Prediction decoder
        self.decoder_pred = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_pred = nn.Linear(hidden_dim, input_dim)

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Encode input sequence. x: (batch, seq_len, input_dim)"""
        output, (h, c) = self.encoder(x)
        return output, (h, c)

    def decode_reconstruct(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Reconstruct input sequence in reverse order."""
        batch_size, seq_len, _ = x.shape

        # Reverse input for teacher forcing during reconstruction
        x_reversed = torch.flip(x, dims=[1])

        output, _ = self.decoder_recon(x_reversed, hidden)
        recon = self.fc_recon(output)
        # Reverse back to original order
        return torch.flip(recon, dims=[1])

    def decode_predict(
        self,
        last_frame: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Predict future frames autoregressively."""
        predictions = []
        current_input = last_frame.unsqueeze(1)  # (batch, 1, input_dim)

        h, c = hidden
        for _ in range(self.pred_len):
            output, (h, c) = self.decoder_pred(current_input, (h, c))
            pred = self.fc_pred(output)  # (batch, 1, input_dim)
            predictions.append(pred)
            current_input = pred  # Autoregressive

        return torch.cat(predictions, dim=1)  # (batch, pred_len, input_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) input skeleton sequence.

        Returns:
            recon: (batch, seq_len, input_dim) reconstructed sequence.
            pred: (batch, pred_len, input_dim) predicted future frames.
        """
        _, hidden = self.encode(x)

        recon = self.decode_reconstruct(x, hidden)
        pred = self.decode_predict(x[:, -1, :], hidden)

        return recon, pred

    def anomaly_score(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        recon_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute per-sample anomaly score.

        Args:
            x: (batch, seq_len, input_dim) input sequence.
            target: (batch, pred_len, input_dim) ground truth future frames.
            recon_weight: Weight for reconstruction vs prediction error.

        Returns:
            scores: (batch,) anomaly scores.
        """
        recon, pred = self.forward(x)

        # Per-sample MSE
        recon_error = ((recon - x) ** 2).mean(dim=(1, 2))
        pred_error = ((pred - target) ** 2).mean(dim=(1, 2))

        return recon_weight * recon_error + (1 - recon_weight) * pred_error
