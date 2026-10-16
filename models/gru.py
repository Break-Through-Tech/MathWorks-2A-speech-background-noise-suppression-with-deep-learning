"""
GRU baseline model for spectrogram-domain speech enhancement.

Architecture (from docs/baseline_model.md):
    input  [B, T, 257]  normalized log-magnitude spectrogram, time-major
    GRU 1  257 → 256 hidden, unidirectional
    GRU 2  256 → 256 hidden, unidirectional, dropout=0.2 between layers
    Linear 256 → 257
    Sigmoid → mask [B, T, 257], values in (0, 1)

Total parameters: ~856 K (~3.4 MB float32).
"""

import torch
import torch.nn as nn


class GRUDenoiser(nn.Module):
    """
    Predict an ideal-ratio mask from a normalized log-magnitude spectrogram.

    The model reads each 16 ms frame left to right (unidirectional) so it
    only needs past context — the same weights could run in real time later.
    """

    def __init__(
        self,
        input_size: int = 257,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 257] normalized log-magnitude, time-major

        Returns:
            mask: [B, T, 257] predicted IRM, values in (0, 1)
        """
        out, _ = self.gru(x)      # [B, T, 256]
        out = self.fc(out)         # [B, T, 257]
        return torch.sigmoid(out)  # [B, T, 257]
