"""Helper functions for visualizing embeddings."""

import time
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import umap
import umap.plot
import wandb


def visualize_embeddings(representations: dict, args: Namespace):
    """Visualize embeddings.

    Args:
        representations (dict): Dictionary of representations datasets.
        args (Namespace): Arguments.
    """
    x_train = representations["train"].data.reshape(-1, 256)
    x_val = representations["val"].data.reshape(-1, 256)
    x_test = representations["test"].data.reshape(-1, 256)
    y_train = representations["train"].labels.reshape(-1)
    y_val = representations["val"].labels.reshape(-1)
    y_test = representations["test"].labels.reshape(-1)

    x_all = x_all = np.concatenate([x_train, x_val, x_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    y_src = np.concatenate(
        [np.full_like(y_train, 0), np.full_like(y_val, 1), np.full_like(y_test, 2)]
    )

    # DEBUG ONLY: subsample everything to 10%
    if args.debug:
        idx_all = np.random.choice(len(y_all), size=int(len(y_all) * 0.1), replace=False)
        x_all = x_all[idx_all]
        y_all = y_all[idx_all]
        y_src = y_src[idx_all]
        idx_train = np.random.choice(len(y_train), size=int(len(y_train) * 0.1), replace=False)
        x_train = x_train[idx_train]
        y_train = y_train[idx_train]
        idx_val = np.random.choice(len(y_val), size=int(len(y_val) * 0.1), replace=False)
        x_val = x_val[idx_val]
        y_val = y_val[idx_val]
        idx_test = np.random.choice(len(y_test), size=int(len(y_test) * 0.1), replace=False)
        x_test = x_test[idx_test]
        y_test = y_test[idx_test]

    mapper = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.2).fit(x_all)

    base_mask = np.zeros_like(y_all, dtype=bool)
    train_mask = base_mask.copy()
    train_mask[: len(y_train)] = True
    val_mask = base_mask.copy()
    val_mask[len(y_train) : len(y_train) + len(y_val)] = True
    test_mask = base_mask.copy()
    test_mask[len(y_train) + len(y_val) :] = True
    all_mask = base_mask.copy()
    all_mask[:] = True

    masks = [train_mask, val_mask, test_mask, all_mask, all_mask]
    ys = [y_all, y_all, y_all, y_all, y_src]
    titles = [
        "UMAP Train",
        "UMAP Val",
        "UMAP Test",
        "UMAP All (action)",
        "UMAP All (source)",
    ]
    for mask, y, title in zip(masks, ys, titles):
        fig, ax = plt.subplots(figsize=(10, 10))
        umap.plot.points(mapper, labels=y, subset_points=mask, ax=ax)
        ax.set_title(title)
        wandb.log({title: wandb.Image(fig)})
        fig.savefig(f"{title}_{time.time()}.pdf", dpi=300)

    ax = umap.plot.connectivity(mapper)
    fig = ax.get_figure()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    ax.set_title("UMAP Connectivity")
    wandb.log({"UMAP Connectivity": wandb.Image(fig)})
    fig.savefig(f"UMAP Connectivity_{time.time()}.pdf", dpi=300)
