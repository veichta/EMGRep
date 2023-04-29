"""Implementation of the InfoNCE loss function for CPC."""

import torch
import torch.nn as nn
from einops import rearrange
from info_nce import InfoNCE


class CPCCriterion(nn.Module):
    """Criterion for CPC model."""

    def __init__(self, k: int):
        """Initialize CPCCriterion.

        Args:
            k (int): Number of steps to look ahead for positive sampling. Must be > 0.
        """
        # TODO: add weights W_k (do not need to be part of model, can be added here)
        super(CPCCriterion, self).__init__()
        assert k > 0, f"k must be > 0, but got {k}"
        self.k = k
        self.infonce = InfoNCE()

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute loss for CPC model.

        ASSUME zSize = cSize <=> W_k === I for all k (same embedding dim, instead of z_{t+k}^TW_kc_t
        like in paper. Conceptually assumes one W for all offsets
        (implcit, GRU invariant down to lin. hom., right?)).

        Args:
            z (torch.Tensor): Encoded batch, shape (batchSize, seqLen, zSize)
            c (torch.Tensor): Autoregressive output, shape (batchSize, seqLen, cSize)

        Returns:
            torch.Tensor: _description_

        Notes:
            - z and c must have the same shape, i.e., zSize == cSize is required.
        """
        """legacy
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
        """

        # (... , cSize, seqLen-k, k-1)
        cPos = torch.cat([c[..., :-k, :] for k in range(1, self.k + 1)], axis=-2)
        cPos = rearrange(cPos, "B P2 L_k EmbC -> (B P2 L_k) EmbC")

        # (..., zSize, seqLen-k, k-1)
        zPos = torch.cat([z[..., k:, :] for k in range(1, self.k + 1)], axis=-2)
        zPos = rearrange(zPos, "B P2 L_k EmbZ -> (B P2 L_k) EmbZ")

        # compute loss
        return self.infonce(cPos, zPos, zPos)
