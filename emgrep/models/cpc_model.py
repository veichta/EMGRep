"""Implementation of CPC model."""

import torch.nn as nn


class CPCEncoder(nn.Module):
    """Encoder for CPC model."""

    def __init__(self, sizeHidden: int = 512):
        """Initialize the encoder.

        Args:
            sizeHidden (int, optional): Size of the hidden layer. Defaults to 512.
        """
        super(CPCEncoder, self).__init__()

        self.module = nn.Sequential(
            nn.Conv1d(10, sizeHidden, 10, stride=5, padding=3),
            nn.BatchNorm1d(sizeHidden),
            nn.ReLU(),
            nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2),
            nn.BatchNorm1d(sizeHidden),
            nn.ReLU(),
            nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1),
            nn.BatchNorm1d(sizeHidden),
            nn.ReLU(),
            nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1),
            nn.BatchNorm1d(sizeHidden),
            nn.ReLU(),
            nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1),
            nn.BatchNorm1d(sizeHidden),
            nn.ReLU(),
        )

    def forward(self, x):
        """Forward pass."""
        x = self.module(x)
        return x


class CPCAR(nn.Module):
    """Autoregressive model for CPC."""

    def __init__(self, dimEncoded: int, dimOutput: int, numLayers: int):
        """Initialize the autoregressive model.

        Args:
            dimEncoded (int): Encoded dimension.
            dimOutput (int): Output dimension.
            numLayers (int): Number of layers.
        """
        super(CPCAR, self).__init__()

        self.gru = nn.GRU(dimEncoded, dimOutput, num_layers=numLayers, batch_first=True)

    def forward(self, x):
        """Encode a batch of sequences."""
        x, _ = self.gru(x)  # discard final hidden state
        return x


class CPCModel(nn.Module):
    """CPC model."""

    def __init__(self, encoder, ar):
        """Initialize the CPC model.

        Args:
            encoder (_type_): _description_
            ar (_type_): _description_
        """
        super(CPCModel, self).__init__()
        self.gEnc = encoder
        self.gAR = ar
        self.gEnc.double()
        self.gAR.double()

    def forward(self, batch):
        """Forward pass."""
        batch = batch.permute(0, 2, 1)  # (batch, channels, time)
        z = self.gEnc(batch)
        z = z.permute(0, 2, 1)  # (batch, time, channels)
        c = self.gAR(z)
        return z, c
