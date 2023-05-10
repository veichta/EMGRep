"""Implementation of CPC model."""

import math

import torch
import torch.nn as nn

# from einops import rearrange
from einops.layers.torch import Rearrange


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
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(kernel_size=3, padding=1, stride=2),
        )

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        conv_type: str = "CAP-CAP",
        backbone_type: str = "ada_avg",
    ):
        """Encoder network for encoding a sequence of blocks into a single vector.

        Args:
            in_channels (int): Number of input channels.
            hidden_dim (int): Feature dimension of the output vector for each block. Will be
            rounded to the next power of 2.
            conv_type (str): which type of conv block to use. One of ["CAP-CAP" (default), "CCAP"]
            backbone_type (str): model on top of convolutional layers (last layers of block encoder)
                One of ["ada_avg" (default), "MLP","attn"]. MLP uses adaptive avg first.
        """
        super().__init__()

        max_power = int(math.log(hidden_dim, 2))
        channels = [in_channels] + [2**f for f in range(5, max_power + 1)]

        conv_block = self._make_conv_block1 if conv_type == "CAP-CAP" else self._make_conv_block2
        self.convs = nn.Sequential(
            *[
                conv_block(in_channels, out_channels)
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ]
        )

        backbone_layers = []
        if backbone_type == "ada_avg":
            # output_conv is used to map the time dimension to a single value
            # -> each block will be mapped to a feature with dimension hidden_dim
            backbone_layers = [nn.AdaptiveAvgPool1d(1)]
        elif backbone_type == "MLP":
            backbone_layers = [
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.1),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.1),
                nn.LeakyReLU(),
            ]
        elif backbone_type == "attn":
            # this only makes sense if time dimension > 1
            encoder_layer = nn.TransformerEncoderLayer(
                batch_first=True, d_model=hidden_dim, nhead=8, dropout=0.15, activation="relu"
            )
            backbone_layers = [
                Rearrange("NKB E L -> NKB L E"),
                nn.TransformerEncoder(encoder_layer, num_layers=3),
                Rearrange("NKB L E -> NKB E L"),
                nn.AdaptiveAvgPool1d(1),
            ]
        else:
            raise NotImplementedError("Unknown backbone type: {}".format(backbone_type))

        self.backbone = nn.Sequential(*backbone_layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, K, num_blocks, block_len, F).
        """
        N, K, num_blocks, block_len, F = x.shape
        x = x.view(N * K * num_blocks, F, block_len)

        x = self.convs(x)
        x = self.backbone(x)

        x = x.view(N, K, num_blocks, -1)

        return x


class CPCAR(nn.Module):
    """Transformer-based Autoregressive model for CPC."""

    def __init__(self, dimEncoded: int, dimOutput: int, numLayers: int, model: str = "GRU"):
        """Initialize the autoregressive model.

        Args:
            dimEncoded (int): Encoded dimension.
            dimOutput (int): Output dimension.
            numLayers (int): Number of layers.
            model (str): Model type. One of ["GRU", "attn"]. Default is GRU like in CPC paper
        """
        super(CPCAR, self).__init__()
        self.model = model
        if self.model == "GRU":
            self.gru = nn.GRU(dimEncoded, dimOutput, num_layers=numLayers, batch_first=True)
            # initialize gru
            for layer_p in self.gru._all_weights:
                for p in layer_p:
                    if "weight" in p:
                        nn.init.kaiming_normal_(
                            self.gru.__getattr__(p), mode="fan_out", nonlinearity="relu"
                        )
        elif self.model == "attn":
            assert dimEncoded == dimOutput
            encoder_layer = nn.TransformerEncoderLayer(
                batch_first=True, d_model=dimEncoded, nhead=8, dropout=0.15, activation="relu"
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=numLayers)
            triu = torch.triu(torch.ones((10, 10)) * float("-inf"), diagonal=1)
            self.causal_mask = torch.nn.Parameter(triu, requires_grad=False)
        else:
            raise NotImplementedError("Unknown model type: {}".format(self.model))

    def forward(self, x):
        """Encode a batch of sequences."""
        N, K, num_blocks, H = x.shape
        x = x.view(N * K, num_blocks, H)
        if self.model == "GRU":
            x, _ = self.gru(x)  # discard final hidden state
        else:
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
