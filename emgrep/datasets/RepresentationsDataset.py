"""Dataset holding the extracted representations."""

import torch
from torch.utils.data import Dataset

from emgrep.models.cpc_model import CPCModel


class RepresentationDataset(Dataset):
    """Dataset holding the extracted representations."""

    def __init__(self, model: CPCModel, dataloader: torch.utils.data.DataLoader) -> None:
        """Initialize the dataset.

        Args:
            model (CPCModel): Model to extract representations from.
            dataloader (torch.utils.data.DataLoader): Dataloader containing the EMG data.
        """
        super().__init__()
        self.data = self._extract_representations(model, dataloader)

    def _extract_representations(
        self, model: CPCModel, dataloader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """Extract representations from the model.

        Args:
            model (CPCModel): Model to extract representations from.
            dataloader (torch.utils.data.DataLoader): Dataloader containing the EMG data.

        Returns:
            torch.Tensor: Extracted representations.
        """
        # TODO: Extract representations
        pass

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return the item at index idx."""
        return self.data[idx]
