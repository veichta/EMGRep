"""Implementation of the training loop for CPC model."""

import datetime
import itertools
import logging
import os
import time
from argparse import Namespace
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

# from torchinfo import summary
from tqdm import tqdm

from emgrep.criterion import CPCCriterion, ExtendedCPCCriterion
from emgrep.models.cpc_model import CPCAR, CPCEncoder, CPCModel


def train_cpc(dataloaders: Dict[str, DataLoader], args: Namespace) -> CPCModel:
    """Train the model.

    Args:
        dataloaders (Dict[str, DataLoader]): Dataloaders for training, validation and testing.
        args (Namespace): Command line arguments.

    Returns:
        CPCModel: Trained model.
    """
    start = time.time()

    # Initialize
    assert args.encoder_dim == args.ar_dim, "Encoder and AR dimensions must be the same for now."

    encoder = CPCEncoder(
        in_channels=16,
        hidden_dim=args.encoder_dim,
        conv_type=args.cpc_conv_type,
        backbone_type=args.cpc_backbone,
    )
    ar = CPCAR(
        dimEncoded=args.encoder_dim,
        dimOutput=args.ar_dim,
        numLayers=args.ar_layers,
        model=args.ar_model,
    )
    cpc_model = CPCModel(encoder=encoder, ar=ar)

    if args.positive_mode == "none":
        criterion = CPCCriterion(k=args.cpc_k, zDim=args.encoder_dim, cDim=args.ar_dim)
    else:
        criterion = ExtendedCPCCriterion(
            k=args.cpc_k,
            zDim=args.encoder_dim,
            cDim=args.ar_dim,
            mode="z",  # TODO: parametrize
            alpha=args.cpc_alpha,
        )

    n_params = sum(p.numel() for p in cpc_model.parameters() if p.requires_grad)
    logging.info(f"Number of trainable parameters: {n_params/1e6:.2f}M")

    if args.wandb:
        wandb.watch(cpc_model, log_freq=100)
        wandb.watch(criterion, log_freq=100)

    # logging.info("Encoder Architecture:")
    # # TODO: parametrize shape
    # summary(encoder, (args.batch_size, 2, 10, 300, 16))
    # logging.info("AR head")
    # summary(ar, (args.batch_size, 1, 10, args.encoder_dim))

    logging.info("Training the model...")

    parameters = itertools.chain(encoder.parameters(), ar.parameters(), criterion.parameters())

    if args.optimizer_cpc == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr_cpc,
            momentum=args.momentum_cpc,
            weight_decay=args.weight_decay_cpc,
        )
    elif args.optimizer_cpc == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=args.lr_cpc,
            weight_decay=args.weight_decay_cpc,
        )
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer_cpc}")

    # Reduce on plateau
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=args.patience_lr_cpc, verbose=True
    )
    # Early stopping init
    best_loss = np.inf
    epochs_no_improve = 0

    # Train model
    metrics: Dict[str, Any] = {"train": {}, "val": {}, "test": {}}
    for epoch in range(args.epochs_cpc):
        metrics["train"][epoch] = train_one_epoch_cpc(
            model=cpc_model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
        )
        metrics["val"][epoch] = validate_cpc(
            model=cpc_model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            epoch=epoch,
            args=args,
        )
        lr_scheduler.step(metrics["val"][epoch]["loss"])

        logging.info(f"Epoch {epoch+1} / {args.epochs_cpc}")
        logging.info(f"Train loss: {metrics['train'][epoch]['loss']:.4f}")
        logging.info(f"Val loss:   {metrics['val'][epoch]['loss']:.4f}")

        save_checkpoint_cpc(model=cpc_model, epoch=epoch, metrics=metrics["val"], args=args)

        if metrics["val"][epoch]["loss"] < best_loss:
            best_loss = metrics["val"][epoch]["loss"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == args.patience_stopping_cpc:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break

    # Test
    metrics["test"] = test(
        model=cpc_model, dataloader=dataloaders["test"], criterion=criterion, epoch=epoch, args=args
    )

    end = time.time()
    elapsed = datetime.timedelta(seconds=end - start)
    logging.info(f"Training time: {elapsed}")

    # Log metrics
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(metrics["train"].keys(), [m["loss"] for m in metrics["train"].values()], label="train")
    ax.plot(metrics["val"].keys(), [m["loss"] for m in metrics["val"].values()], label="val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()

    plt.savefig(os.path.join(args.log_dir, "cpc_loss.png"))
    fig.clear()

    return cpc_model


def train_one_epoch_cpc(
    model: CPCModel,
    dataloader: DataLoader,
    criterion: Union[CPCCriterion, ExtendedCPCCriterion],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: Namespace,
) -> Dict[str, float]:
    """Train the model for one epoch.

    Args:
        model (CPCModel): The model to train.
        dataloader (DataLoader): The training dataloader.
        criterion (CPCCriterion): The CPC loss criterion.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): The current epoch.
        args (Namespace): Command line arguments.

    Returns:
        Dict[str, float]: Training metrics.
    """
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1} / {args.epochs_cpc}", ncols=100)
    losses = []
    model.train()
    model.to(args.device)
    criterion.to(args.device)
    for emg, stimulus, _ in pbar:
        optimizer.zero_grad()
        out = model(emg.to(args.device))
        loss = criterion(*out, stimulus.to(args.device))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if args.wandb:
            wandb.log({"train_loss": loss.item()})

        pbar.set_postfix({"loss": np.mean(losses)})

    return {"loss": np.mean(losses)}


def validate_cpc(
    model: CPCModel,
    dataloader: DataLoader,
    criterion: Union[CPCCriterion, ExtendedCPCCriterion],
    epoch: int,
    args: Namespace,
) -> Dict[str, float]:
    """Validate the model.

    Args:
        model (CPCModel): The model to validate.
        dataloader (DataLoader): The validation dataloader.
        criterion (CPCCriterion): The CPC loss criterion.
        epoch (int): The current epoch.
        args (Namespace): Command line arguments.

    Returns:
        Dict[str, float]: Validation metrics.
    """
    pbar = tqdm(dataloader, desc=f"Valid Epoch {epoch+1} / {args.epochs_cpc}", ncols=100)

    losses = []
    model.to(args.device)
    criterion.to(args.device)
    model.eval()
    criterion.eval()
    with torch.no_grad():
        for emg, stimulus, _ in pbar:
            out = model(emg.to(args.device))

            loss = criterion(*out, stimulus.to(args.device))
            losses.append(loss.item())

            if args.wandb:
                wandb.log({"val_loss": loss.item()})

            pbar.set_postfix({"loss": np.mean(losses)})
    return {"loss": np.mean(losses)}


def test(
    model: CPCModel,
    dataloader: DataLoader,
    criterion: CPCCriterion,
    epoch: int,
    args: Namespace,
) -> Dict[str, float]:
    """Test the model.

    Args:
        model (CPCModel): The model to test.
        dataloader (DataLoader): The test dataloader.
        criterion (CPCCriterion): The CPC loss criterion.
        epoch (int): The current epoch.
        args (Namespace): Command line arguments.

    Returns:
        Dict[str, float]: Test metrics.
    """
    pbar = tqdm(dataloader, desc=f"Testing {epoch+1} / {args.epochs_cpc}", ncols=100)
    losses = []
    model.to(args.device)
    criterion.to(args.device)
    model.eval()
    criterion.eval()
    with torch.no_grad():
        for emg, stimulus, _ in pbar:
            out = model(emg.to(args.device))
            loss = criterion(*out, stimulus.to(args.device))
            losses.append(loss.item())

            pbar.set_postfix({"loss": np.mean(losses)})

    if args.wandb:
        wandb.log({"test_loss": np.mean(losses)})

    return {"loss": np.mean(losses)}


def save_checkpoint_cpc(
    model: CPCModel,
    epoch: int,
    args: Namespace,
    metrics: Dict[str, Any],
) -> None:
    """Save the model checkpoint."""
    model_dir = os.path.join(args.log_dir, "checkpoints")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    losses = [m["loss"] for m in metrics.values()]
    best_epoch = np.argmin(losses)

    if epoch == best_epoch:
        logging.info(f"Saving model checkpoint at epoch {epoch+1}...")
        torch.save(model.to("cpu").state_dict(), os.path.join(model_dir, "best_model.pt"))
        model.to(args.device)
