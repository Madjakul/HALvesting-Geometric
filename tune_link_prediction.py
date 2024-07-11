# tune_link_prediction.py
# TODO: Use Ray Tune to perform hyperparameter tuning
# TODO: https://medium.com/distributed-computing-with-ray/scaling-up-pytorch-lightning-hyperparameter-tuning-with-ray-tune-4bd9e1ff9929
# TODO: https://docs.ray.io/en/latest/tune/index.html

import logging
from typing import Any, Dict

import lightning as L
import psutil
import torch
import torch_geometric.transforms as T
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import LinkNeighborLoader

from halvesting_geometric.modules import LinkPrediction
from halvesting_geometric.utils import helpers, logging_config
from halvesting_geometric.utils.argparsers import LinkPredictionArgparse
from halvesting_geometric.utils.data import LinkPredictionDataset

logging_config()


def train_tune(config: Dict[str, Any]):
    edge_label_index = train_data["author", "writes", "paper"].edge_label_index
    edge_label = train_data["author", "writes", "paper"].edge_label
    train_dataloader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=config["num_neighbors"],
        neg_sampling_ratio=config["neg_sampling_ratio"],
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label=edge_label,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_proc,
        persistent_workers=True,
    )
    edge_label_index = val_data["author", "writes", "paper"].edge_label_index
    edge_label = val_data["author", "writes", "paper"].edge_label
    val_dataloader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=config["num_neighbors"],
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label=edge_label,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_proc,
        persistent_workers=True,
    )
    edge_label_index = test_data["author", "writes", "paper"].edge_label_index
    edge_label = test_data["author", "writes", "paper"].edge_label
    test_dataloader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=config["num_neighbors"],
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label=edge_label,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_proc,
        persistent_workers=True,
    )

    logging.info("Creating model...")
    model = LinkPrediction(
        gnn=args.gnn,  # type: ignore
        metadata=data.metadata(),
        paper_num_nodes=data["paper"].num_nodes,
        author_num_nodes=data["author"].num_nodes,
        institution_num_nodes=data["institution"].num_nodes,
        domain_num_nodes=data["domain"].num_nodes,
        hidden_channels=config["hidden_channels"],
        dropout=config["dropout"],
    )

    logging.info("Training and testing model...")
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=-1,
        logger=wandb_logger if args.wandb else None,
        max_epochs=config["max_epochs"],
    )
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    trainer.test(model=model, dataloaders=test_dataloader)


if __name__ == "__main__":
    train_tune()
