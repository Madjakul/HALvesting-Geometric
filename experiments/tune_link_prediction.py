# experiments/tune_link_prediction.py

import logging
from functools import partial
from typing import Any, Dict

import lightning as L
import psutil
import torch
import torch_geometric.transforms as T
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from halvesting_geometric.modules import LinkPrediction
from halvesting_geometric.utils import logging_config
from halvesting_geometric.utils.argparsers import TuneLinkPredictionArgparse
from halvesting_geometric.utils.data import LinkPredictionDataset

GNN = "sage"


logging_config()


def train_tune(
    config: Dict[str, Any],
    gnn: str,
    callback: TuneReportCallback,
    data: HeteroData,
    train_data: HeteroData,
    val_data: HeteroData,
    num_proc: int,
    accelerator: str,
):
    edge_label_index = train_data["author", "writes", "paper"].edge_label_index
    edge_label = train_data["author", "writes", "paper"].edge_label
    train_dataloader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=config["num_neighbors"] * 2,
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label=edge_label,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_proc,
        persistent_workers=True,
    )
    edge_label_index = val_data["author", "writes", "paper"].edge_label_index
    edge_label = val_data["author", "writes", "paper"].edge_label
    val_dataloader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=config["num_neighbors"] * 2,
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label=edge_label,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_proc,
        persistent_workers=True,
    )

    logging.info("Creating model...")
    model = LinkPrediction(
        gnn=gnn,  # type: ignore
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
        accelerator=accelerator, devices=-1, max_epochs=10, callbacks=[callback]
    )
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    args = TuneLinkPredictionArgparse.parse_known_args()

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

    dataset = LinkPredictionDataset(args.root_dir, lang=args.lang_)
    data = dataset[0]
    data = T.ToUndirected()(data)

    logging.info("Creating train, validation, and test datasets...")
    transform = T.RandomLinkSplit(
        num_val=0.3,
        num_test=0,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=True,
        edge_types=("author", "writes", "paper"),
        rev_edge_types=("paper", "rev_writes", "author"),
    )
    train_data, val_data, test_data = transform(data)
    callback = TuneReportCallback(
        {"loss": "val_loss", "roc_auc": "val_roc_auc"}, on="validation_end"
    )

    config = {
        "num_neighbors": tune.choice([16, 32, 64, 128]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "hidden_channels": tune.choice([16, 32, 64, 128]),
        "dropout": tune.uniform(0.1, 0.5),
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-4, 0.0),
    }
    tune.run(
        partial(
            train_tune,
            gnn=GNN,
            callback=callback,
            data=data,
            train_data=train_data,
            val_data=val_data,
            num_proc=num_proc,
            accelerator=accelerator,
        ),
        metric="roc_auc",
        mode="max",
        name=f"Tune Link Prediction {GNN}",
        resources_per_trial={"cpu": num_proc, "gpu": 1},
        num_samples=10,
        search_alg=BayesOptSearch(),
        scheduler=AsyncHyperBandScheduler(),
        config=config,
    )
