"""Dataset holding the extracted representations."""

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
        self.data, self.labels = self._extract_representations(model, dataloader)

        try:
            import numpy as np

            vals = np.array([x.numpy() for x in self.labels]).flatten().astype(np.int8)
            print(np.unique(vals))
        except:
            # WTF?
            print("Heterogeneous labels received from dataloader")
            pass

    def _extract_representations(
        self, model: CPCModel, dataloader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """Extract representations from the model. That is, returns data and labels for the first pair item for each pair

        Args:
            model (CPCModel): Model to extract representations from.
            dataloader (torch.utils.data.DataLoader): Dataloader containing the EMG data.

        Returns:
            tuple: (torch.Tensor, torch.tensor): Extracted representations.
        """
        # @TODO test
        # @TODO where do we add the labels? Do they need resampling?
        with torch.no_grad():
            return zip(
                *(
                    (model(x)[1][:, 0], y[:, 0, :, 0, 0])
                    for x, y, _ in tqdm(dataloader, desc="Generating Embeddings")
                )
            )  # takes the 2nd output of the model (which is c) and the first sample of all pairs as DATA and the first label of the first sample of label pairs as label (one lbl per block)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return the item at index idx."""
        return self.data[idx], self.labels[idx]
