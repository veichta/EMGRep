"""Functionality for different autoregressive models."""
import torch
from torch import nn


class LSTMAR(nn.Module):
    """Autoregressive model for CPC."""

    def __init__(self, dimEncoded: int, dimOutput: int, numLayers: int):
        """Initialize the autoregressive model.

        Args:
            dimEncoded (int): Encoded dimension.
            dimOutput (int): Output dimension.
            numLayers (int): Number of layers.
        """
        super(LSTMAR, self).__init__()

        self.lstms = [
            nn.LSTM(dimEncoded, dimOutput, num_layers=numLayers, batch_first=True, dropout=0.15)
        ]
        self.lstms += [
            nn.LSTM(dimOutput, dimOutput, num_layers=numLayers, batch_first=True, dropout=0.15)
            for _ in range(numLayers - 1)
        ]

    def forward(self, x):
        """Encode a batch of sequences."""
        N, K, num_blocks, H = x.shape
        x = x.view(N * K, num_blocks, H)
        for lstm in self.lstms:
            x, _ = lstm(x)

        x = x.view(N, K, num_blocks, -1)
        return x


class TransformerAR(nn.Module):
    """Transformer/GRU-based Autoregressive model for CPC."""

    def __init__(self, dimEncoded: int, dimOutput: int, numLayers: int):
        """Initialize the autoregressive model.

        Args:
            dimEncoded (int): Encoded dimension.
            dimOutput (int): Output dimension.
            numLayers (int): Number of layers.
        """
        super(TransformerAR, self).__init__()

        if dimEncoded != dimOutput:
            self.linear = torch.nn.Linear(dimEncoded, dimOutput)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            batch_first=True, d_model=dimOutput, nhead=8, dropout=0.15, activation="relu"
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=numLayers)

    def forward(self, x):
        """Encode a batch of sequences."""
        N, K, num_blocks, H = x.shape
        x = x.view(N * K, num_blocks, H)
        if self.linear is not None:
            x = self.linear(x)
        causal_mask = torch.triu(torch.ones((num_blocks, num_blocks)) * float("-inf"), diagonal=1)
        x = self.transformer_encoder(src=x, mask=causal_mask)
        x = x.view(N, K, num_blocks, -1)
        return x
