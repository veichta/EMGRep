"""Implementation of CPC model."""

import math

import torch.nn as nn


class CPCEncoder(nn.Module):
    """Encoder network for CPC."""

    def __init__(self, in_channels: int, hidden_dim: int):
        """Encoder network for encoding a sequence of blocks into a single vector.

        Args:
            in_channels (int): Number of input channels.
            hidden_dim (int): Feature dimension of the output vector for each block. Will be
            rounded to the next power of 2.
        """
        super().__init__()

        max_power = int(math.log(hidden_dim, 2))
        result = [in_channels] + [2**f for f in range(5, max_power + 1)]

        self.convs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(in_channels),
                    nn.MaxPool1d(kernel_size=2, padding=1),
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels),
                    nn.MaxPool1d(kernel_size=2, padding=1),
                )
                for in_channels, out_channels in zip(result[:-1], result[1:])
            ]
        )

        # output_conv is used to map the time dimension to a single value
        # -> each block will be mapped to a feature with dimension hidden_dim
        self.output_conv = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, K, num_blocks, block_len, F).
        """
        N, K, num_blocks, block_len, F = x.shape
        x = x.view(N * K * num_blocks, F, block_len)

        x = self.convs(x)
        x = self.output_conv(x)

        x = x.view(N, K, num_blocks, -1)

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
        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if "weight" in p:
                    nn.init.kaiming_normal_(
                        self.gru.__getattr__(p), mode="fan_out", nonlinearity="relu"
                    )

    def forward(self, x):
        """Encode a batch of sequences."""
        N, K, num_blocks, H = x.shape
        x = x.view(N * K, num_blocks, H)
        x, _ = self.gru(x)  # discard final hidden state

        x = x.view(N, K, num_blocks, -1)
        return x


class CPCModel(nn.Module):
    """CPC model."""

    def __init__(self, encoder: CPCEncoder, ar: CPCAR):
        """Initialize the CPC model.

        Args:
            encoder (_type_): _description_
            ar (_type_): _description_
        """
        super(CPCModel, self).__init__()
        self.gEnc = encoder
        self.gAR = ar

    def forward(self, batch):
        """Forward pass."""
        z = self.gEnc(batch)
        c = self.gAR(z)
        return z, c
