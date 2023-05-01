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
        assert (
            zSeqLen == cSeqLen
        ), f"z and c must have same sequence length, but got {zSeqLen} and {cSeqLen} respectively"
        assert zDim == self.zDim, f"zDim must be {self.zDim}, but got {zDim}"
        assert cDim == self.cDim, f"cDim must be {self.cDim}, but got {cDim}"

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
        return self.infonce(anchor, positive, negative)


class ExtendedCPCCriterion(nn.Module):
    """Our Extension of the criterion for the CPC model.

    We add an additional loss term by computing the InfoNCE loss for the blocks based on the given
    pairings from the dataloader.
    """

    def __init__(self, k: int, zDim: int = 256, cDim: int = 256, mode="z", alpha=0.5):
        """Initialize ExtendedCriterion.

        Args:
            k (int): Number of steps to look ahead for positive sampling. Must be > 0.
            zDim (int, optional): Dimension of encoder output. Defaults to 256.
            cDim (int, optional): Dimension of autoregressive model output. Defaults to 256.
            mode (str, optional): Mode for the loss computation. Defaults to "z". Can be "z" or "c".
            alpha (float, optional): Weight for the additional loss term. Defaults to 0.5.
        """
        super(ExtendedCPCCriterion, self).__init__()
        assert k > 0, f"k must be > 0, but got {k}"
        self.k = k
        self.zDim = zDim
        self.cDim = cDim
        assert mode in ["z", "c"], f"mode must be 'z' or 'c', but got {mode}"
        self.mode = mode
        self.alpha = alpha
        self.infonce = InfoNCE()
        self.canonical_criterion = CPCCriterion(k, zDim, cDim)
        self.linear = nn.Linear(zDim, zDim)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute loss for CPC model.

        Args:
            z (torch.Tensor): Encoded batch, shape (batchSize, 2, zDim, seqLen)
            c (torch.Tensor): Autoregressive output, shape (batchSize, 2, cDim, seqLen)

        Returns:
            torch.Tensor: Loss for CPC model.
        """
        assert z.shape[1] == 2, f"z's second dimension must have size 2, but got {z.shape[1]}"
        assert c.shape[1] == 2, f"c's second dimension must have size 2, but got {c.shape[1]}"

        if self.mode == "z":
            anchor = z[:, 0, :, :].flatten(0, 1)
            positive = self.ext_linear(z[:, 1, :, :].flatten(0, 1))
        elif self.mode == "c":
            anchor = c[:, 0, :, :].flatten(0, 1)
            positive = self.ext_linear(c[:, 1, :, :].flatten(0, 1))
        else:
            raise ValueError(f"mode must be 'z' or 'c', but got {self.mode}")
        ext_loss = self.infonce(anchor, positive)

        return self.canonical_criterion(z, c) + self.alpha * ext_loss
