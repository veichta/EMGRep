"""Implementation of a TCN encoder similar to the one in the thesis."""
import math

import torch
import torch.nn as nn


class TCNEncoder(nn.Module):
    """TCN encoder like the one im the thesis."""

    def __init__(
        self, block_len: int, in_channels: int = 16, hidden_dim: int = 128, max_channels=64
    ):
        """Initialize the dummy encoder.

        Args:
            block_len (int): Length of input sequence to scale receptive field to.
            in_channels (int, optional): Number of input channels. Defaults to 16.
            hidden_dim (int, optional): Number of output channels. Defaults to 128.
            max_channels (int, optional): Maximum number of channels in TCN. Defaults to 64.
        """
        super(TCNEncoder, self).__init__()
        max_power = int(math.ceil(math.log(block_len, 2)))
        num_channels = [min(max_channels, 2**f) for f in range(5, max_power + 1)]
        print("CHANNELS: ", num_channels)
        self.tcn_module = TemporalConvNet(in_channels, num_channels)
        self.to_out = torch.nn.Sequential(
            *[
                # torch.nn.BatchNorm1d(num_channels[-1]),
                torch.nn.Linear(num_channels[-1], hidden_dim),
                # torch.nn.ELU(),
                # torch.nn.Dropout(0.4),
                # torch.nn.BatchNorm1d(n_mlp),
                # torch.nn.Linear(n_mlp, hidden_dim)
            ]
        )

    def forward(self, x):
        """Forward pass."""
        N, K, num_blocks, block_len, F = x.shape
        x = x.view(N * K * num_blocks, F, block_len)

        x = self.tcn_module(x)
        x = self.to_out(x[:, :, -1])

        x = x.view(N, K, num_blocks, -1)
        return x


class Chomp1d(nn.Module):
    """Makes the convolution causal.

    source: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """

    def __init__(self, chomp_size):
        """Initialize the module."""
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """Forward pass."""
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A TCN residual block.

    source: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """Initialize the module."""
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ELU()
        # self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """Forward pass."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """TCN module.

    source: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """Initialize the module."""
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)
