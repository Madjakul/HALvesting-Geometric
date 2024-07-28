# link_prediction.py

import logging

import lightning as L
import psutil
import torch
import torch_geometric.transforms as T
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from halvesting_geometric.modules import LinkPrediction
from halvesting_geometric.utils import helpers, logging_config
from halvesting_geometric.utils.argparsers import LinkPredictionArgparse
from halvesting_geometric.utils.data import LinkPredictionDataModule

METADATA = (
    ["paper", "author", "affiliation", "domain"],
    [
        ("author", "writes", "paper"),
        ("author", "affiliated_with", "affiliation"),
        ("paper", "cites", "paper"),
        ("paper", "has_topic", "domain"),
        ("paper", "rev_writes", "author"),
        ("affiliation", "rev_affiliated_with", "author"),
        ("domain", "rev_has_topic", "paper"),
    ],
)

logging_config()


if __name__ == "__main__":
    args = LinkPredictionArgparse.parse_known_args()

    try:
        torch.set_float32_matmul_precision("medium")
    except Exception as e:
        logging.error(f"Unable to activate TensorCore:\n{e}")

    config = helpers.load_config_from_file(args.config_file)
    logging.info(f"Configuration file loaded from {args.config_file}.")

    num_proc = (
        psutil.cpu_count(logical=False) if args.num_proc is None else args.num_proc
    ) - 2
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

    dataset = LinkPredictionDataModule(
        data_dir=args.root_dir,
        batch_size=config["batch_size"],
        num_neighbors=config["num_neighbors"],
        lang=args.lang_,
        neg_sampling_ratio=config["neg_sampling_ratio"],
        num_proc=num_proc,
        num_val=config["num_val"],
        num_test=config["num_test"],
        add_negative_train_samples=True,
        shuffle_train=True,
        shuffle_val=False,
        persistent_workers=True,
    )

    logging.info("Creating model...")
    model = LinkPrediction(
        gnn=config["gnn"],
        metadata=METADATA,
        batch_size=config["batch_size"],
        author_num_nodes=config["author_num_nodes"],
        institution_num_nodes=config["affiliation_num_nodes"],
        domain_num_nodes=config["domain_num_nodes"],
        hidden_channels=config["hidden_channels"],
        dropout=config["dropout"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    logging.info("Training and testing model...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.project_root}/tmp", save_top_k=2, monitor="val_roc_auc"
    )
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=-1,
        logger=wandb_logger if args.wandb else None,
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, datamodule=dataset)
    trainer.test(model, datamodule=dataset)
