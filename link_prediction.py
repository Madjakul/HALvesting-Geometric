# link_prediction.py

import logging

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


if __name__ == "__main__":
    args = LinkPredictionArgparse.parse_known_args()

    config = helpers.load_config_from_file(args.config_file)
    logging.info(f"Configuration file loaded from {args.config_file}.")

    num_proc = (
        psutil.cpu_count(logical=False) if args.num_proc is None else args.num_proc
    )
    logging.info(f"Number of processes: {num_proc}.")

    if args.accelerator is None:
        if torch.cuda.is_available():
            accelerator = "gpu"
        else:
            accelerator = "cpu"
    else:
        accelerator = args.accelerator
    logging.info(f"Accelerator: {accelerator}.")

    if args.wandb:
        wandb_logger = WandbLogger(
            project="HALvest-Geometric",
            name=f"link-prediction-{config['gnn']}-{args.run}",
        )
        logging.info("WandB logger enabled.")

    dataset = LinkPredictionDataset(args.root_dir, lang=args.lang_)
    data = dataset[0]
    data = T.ToUndirected()(data)

    logging.info("Creating train, validation, and test datasets...")
    transform = T.RandomLinkSplit(
        num_val=config["num_val"],
        num_test=config["num_test"],
        neg_sampling_ratio=config["neg_sampling_ratio"],
        add_negative_train_samples=False,
        edge_types=("author", "writes", "paper"),
        rev_edge_types=("paper", "rev_writes", "author"),
    )
    train_data, val_data, test_data = transform(data)

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
        lr=config["lr"],
        weight_decay=config["weight_decay"],
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
