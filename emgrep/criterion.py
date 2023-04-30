"""Implementation of the infoNCE loss function for CPC."""

import torch
import torch.nn as nn
from info_nce import InfoNCE


class CPCCriterion(nn.Module):
    """Criterion for CPC model."""

    def __init__(self, k: int, zDim: int = 256, cDim: int = 256):
        """Initialize CPCCriterion.

        Args:
            k (int): Number of steps to look ahead for positive sampling. Must be > 0.
            zDim (int, optional): Dimension of encoder output. Defaults to 256.
            cDim (int, optional): Dimension of autoregressive model output. Defaults to 256.
        """
        super(CPCCriterion, self).__init__()
        assert k > 0, f"k must be > 0, but got {k}"
        self.k = k
        self.zDim = zDim
        self.cDim = cDim
        self.infonce = InfoNCE()
        self.linear = nn.ModuleList([nn.Linear(cDim, zDim) for _ in range(k)])

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute loss for CPC model.

        Args:
            z (torch.Tensor): Encoded batch, shape (batchSize, 1 (or 2), zDim, seqLen)
            c (torch.Tensor): Autoregressive output, shape (batchSize, 1 (or 2), cDim, seqLen)

        Returns:
            torch.Tensor: Loss for CPC model.
        """
        zBatchSize, _, zSeqLen, zDim = z.shape
        cBatchSize, _, cSeqLen, cDim = c.shape

        assert zBatchSize == cBatchSize, f"batchSize must be {zBatchSize}, but got {cBatchSize}"
        assert zDim == self.zDim, f"zDim must be {self.zDim}, but got {zDim}"
        assert cDim == self.cDim, f"cDim must be {self.cDim}, but got {cDim}"
        assert (
            zSeqLen == cSeqLen
        ), f"z and c must have same sequence length, but got {zSeqLen} and {cSeqLen} respectively"

        # compute anchor and positive samples
        anchor, positive = [], []
        for offset in range(1, self.k + 1):
            cPos = c[:, 0, :-offset, :].flatten(0, 1)
            cPos = self.linear[offset - 1](cPos)
            anchor.append(cPos)
            zPos = z[:, 0, offset:, :].flatten(0, 1)
            positive.append(zPos)
        anchor = torch.cat(anchor, dim=0)  # (batchSize * (seqLen-k), zDim)
        positive = torch.cat(positive, dim=0)  # (batchSize * (seqLen-k), zDim)
        negative = z[:, 0, :, :].flatten(0, 1)  # (batchSize * seqLen, zDim)

        # compute loss
        return self.infonce(positive, anchor, negative)  # self.infonce(anchor, positive, negative)
