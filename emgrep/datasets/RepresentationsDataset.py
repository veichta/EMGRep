"""Dataset holding the extracted representations."""

import logging
from argparse import Namespace

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from emgrep.models.cpc_model import CPCModel


class RepresentationDataset(Dataset):
    """Dataset holding the extracted representations."""

    def __init__(
        self, model: CPCModel, dataloader: torch.utils.data.DataLoader, args: Namespace
    ) -> None:
        """Initialize the dataset.

        Args:
            model (CPCModel): Model to extract representations from.
            dataloader (torch.utils.data.DataLoader): Dataloader containing the EMG data.
            args (Namespace): Command line arguments.
        """
        super().__init__()
        self.data, self.labels = self._extract_representations(model, dataloader, args)
        logging.debug(f"Data shape:   {self.data.shape}")
        logging.debug(f"Labels shape: {self.labels.shape}")

        # map labels to 0, 1, 2, ...
        self.actual_labels = torch.unique(self.labels).to(torch.int64).numpy()
        logging.debug(
            f"Mapping labels {list(self.actual_labels)} to {list(range(len(self.actual_labels)))}"
        )
        self.label_map = {label: i for i, label in enumerate(self.actual_labels)}
        self.labels = torch.tensor(
            [[self.label_map[c.item()] for c in labels] for labels in self.labels]
        )

        # try:
        #     import numpy as np

        #     vals = np.array([x.numpy() for x in self.labels]).flatten().astype(np.int8)
        #     logging.warning(np.unique(vals))
        # except Exception:
        #     # WTF?
        #     logging.warning("Heterogeneous labels received from dataloader")

    def _extract_representations(
        self, model: CPCModel, dataloader: torch.utils.data.DataLoader, args: Namespace
    ) -> torch.Tensor:
        """Extract representations from the model.

        That is, returns data and labels for the first pair item for each pair (i.e. the first
        sample of each block)

        Args:
            model (CPCModel): Model to extract representations from.
            dataloader (torch.utils.data.DataLoader): Dataloader containing the EMG data.
            args (Namespace): Command line arguments.

        Returns:
            torch.Tensor: Extracted representations.
        """
        # @TODO test
        # @TODO where do we add the labels? Do they need resampling?
        model.to(args.device)
        model.eval()
        with torch.no_grad():
            representations = []
            labels = []
            for x, y, _ in tqdm(dataloader, desc="Generating Embeddings", ncols=100):
                representation = model(x.to(args.device))[1][:, 0, :]
                label = y[:, 0, :, -1, 0]
                representations.append(representation)
                labels.append(label)

            return torch.cat(representations), torch.cat(labels).to(torch.int64)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return the item at index idx."""
        return self.data[idx], self.labels[idx]
