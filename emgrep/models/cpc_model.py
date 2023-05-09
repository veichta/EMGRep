"""Implementation of CPC model."""

import math

import torch
import torch.nn as nn
from einops import rearrange

from emgrep.models.tcn import TemporalConvNet


class CPCEncoder(nn.Module):
    """Encoder network for CPC."""

    def _make_conv_block1(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(in_channels),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(kernel_size=2, padding=1),
        )

    def _make_conv_block2(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(0.15),
            nn.ELU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(kernel_size=3, padding=1, stride=2),
        )

    def __init__(self, in_channels: int, hidden_dim: int):
        """Encoder network for encoding a sequence of blocks into a single vector.

        Args:
            in_channels (int): Number of input channels.
            hidden_dim (int): Feature dimension of the output vector for each block. Will be
            rounded to the next power of 2.
        """
        super().__init__()

        max_power = int(math.log(hidden_dim, 2))
        channels = [in_channels] + [2**f for f in range(5, max_power + 1)]

        self.convs = nn.Sequential(
            *[
                self._make_conv_block1(in_channels, out_channels)
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ]
        )

        if False:
            encoder_layer = nn.TransformerEncoderLayer(
                batch_first=True, d_model=hidden_dim, nhead=8, dropout=0.15, activation="gelu"
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=5)

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

        if False:
            x = rearrange(x, "NKB E L -> NKB L E")
            x = self.transformer_encoder(x)
            x = rearrange(x, "NKB L E -> NKB E L")

        x = self.output_conv(x)

        x = x.view(N, K, num_blocks, -1)

        return x


class _CPCEncoder(nn.Module):
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

        # 64, 2, 10, 300, 16
        self.TCN = TemporalConvNet(in_channels, num_channels=[16, 32, 64, 16])
        self.project = nn.Linear(in_channels, 16)
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
        x = rearrange(x, "N K B Ln Ch -> (N K) Ch (B Ln)")
        # x = x.view(N * K * num_blocks, F, block_len)
        res = self.project(rearrange(x, "NK Ch BLn -> NK BLn Ch"))
        res = rearrange(res, "NK BLn Ch-> NK Ch BLn")
        x = self.TCN(x)
        x = x + res
        x = rearrange(x, "X Ch (B Ln) -> (X B) Ch Ln", B=num_blocks)
        x = self.convs(x)
        x = self.output_conv(x)
        # x = rearrange(x, "(N K) E B -> N K B E)", N=N)
        x = x.view(N, K, num_blocks, -1)

        return x


class _CPCAR(nn.Module):
    """Autoregressive model for CPC."""

    def __init__(self, dimEncoded: int, dimOutput: int, numLayers: int):
        """Initialize the autoregressive model.

        Args:
            dimEncoded (int): Encoded dimension.
            dimOutput (int): Output dimension.
            numLayers (int): Number of layers.
        """
        super(_CPCAR, self).__init__()

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


class CPCAR(nn.Module):
    """Transformer-based Autoregressive model for CPC."""

    def __init__(self, dimEncoded: int, dimOutput: int, numLayers: int):
        """Initialize the autoregressive model.

        Args:
            dimEncoded (int): Encoded dimension.
            dimOutput (int): Output dimension.
            numLayers (int): Number of layers.
        """
        super(CPCAR, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True, d_model=dimEncoded, nhead=8, dropout=0.15, activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=numLayers)
        triu = torch.triu(torch.ones((10, 10)) * float("-inf"), diagonal=1)
        self.causal_mask = torch.nn.Parameter(triu, requires_grad=False)

    def forward(self, x):
        """Encode a batch of sequences."""
        N, K, num_blocks, H = x.shape
        x = x.view(N * K, num_blocks, H)
        x = self.transformer_encoder(src=x, mask=self.causal_mask)
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
