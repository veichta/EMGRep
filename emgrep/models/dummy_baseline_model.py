"""Implementation of a dummy model for the baseline."""

import torch
import torch.nn as nn


class DummyBaselineEncoder(nn.Module):
    """Dummy encoder for the baseline model."""

    def __init__(self, in_channels: int = 16, out_channels: int = 128):
        """Initialize the dummy encoder.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 16.
            out_channels (int, optional): Number of output channels. Defaults to 128.
        """
        super(DummyBaselineEncoder, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(in_channels, 128, 10, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(10),
            nn.Conv1d(128, 128, 10, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(10),
            nn.Conv1d(128, 128, 3, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )

    def forward(self, x):
        """Forward pass."""
        x = self.module(x)
        return x


class DummyBaselineAR(nn.Module):
    """Dummy autoregressive model for the baseline model."""

    def __init__(self, dimEncoded: int = 128, dimOutput: int = 128, numLayers: int = 1):
        """Initialize the dummy autoregressive model."""
        super(DummyBaselineAR, self).__init__()

        self.gru = nn.GRU(dimEncoded, dimOutput, num_layers=numLayers, batch_first=True)

    def forward(self, x):
        """Encode a batch of sequences."""
        x, _ = self.gru(x)  # discard final hidden state
        return x


class DummyBaselineModel(nn.Module):
    """Dummy model for the baseline."""

    def __init__(self, encoder, ar):
        """Initialize the DummyBaselineModel."""
        super(DummyBaselineModel, self).__init__()
        self.gEnc = encoder
        self.gAR = ar
        self.gEnc.double()
        self.gAR.double()

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x (torch.Tensor): EMG data input. Expected to be of shape
                (batch_size, 1, num_blocks, block_size, channels)


        """
        x = x[:, 0, :, :, :]
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        x = x.permute(0, 2, 1)  # (batch, channels, time)
        z = self.gEnc(x)
        z = z.permute(0, 2, 1)  # (batch, time, channels)
        c = self.gAR(z)
        return z, c
