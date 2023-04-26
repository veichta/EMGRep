"""Implementation of the infoNCE loss function for CPC."""

import torch
import torch.nn as nn
from info_nce import InfoNCE


class CPCCriterion(nn.Module):
    """Criterion for CPC model."""

    def __init__(self, k: int):
        """Initialize CPCCriterion.

        Args:
            k (int): Number of steps to look ahead for positive sampling.
        """
        super(CPCCriterion, self).__init__()
        self.k = k
        self.infonce = InfoNCE()

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute loss for CPC model.

        Args:
            z (torch.Tensor): Encoded batch, shape (batchSize, zSize, seqLen)
            c (torch.Tensor): Autoregressive output, shape (batchSize, cSize, seqLen)

        Returns:
            torch.Tensor: _description_
        """
        zBatchSize, zDim, zSeqLen = z.shape
        cBatchSize, cDim, cSeqLen = c.shape
        assert (
            zBatchSize == cBatchSize and zSeqLen == cSeqLen and zDim == cDim
        ), f"z and c must have the same shape, but got z: {z.shape} and c: {c.shape}"

        # positive sampling: take pairs of z and c with distance k
        cPos = c[:, :, : -self.k]  # (batchSize, cSize, seqLen-k)
        cPos = cPos.permute(0, 2, 1)  # (batchSize, seqLen-k, cSize)
        cPos = cPos.reshape(-1, cDim)  # (batchSize * (seqLen-k), cSize)
        zPos = z[:, :, self.k :]  # (batchSize, zSize, seqLen-k)
        zPos = zPos.permute(0, 2, 1)  # (batchSize, seqLen-k, zSize)
        zPos = zPos.reshape(-1, zDim)  # (batchSize * (seqLen-k), zSize)

        # negative sampling: shuffle z vectors
        zNeg = z[:, :, self.k :]  # (batchSize, zSize, seqLen-k)
        zNeg = zNeg.permute(0, 2, 1)  # (batchSize, seqLen-k, zSize)
        zNeg = zNeg.reshape(-1, zDim)  # (batchSize * (seqLen-k), zSize)
        zNeg = zNeg[torch.randperm(zNeg.shape[0])]  # shuffle zNeg

        # compute loss
        return self.infonce(zPos, cPos, zNeg)
