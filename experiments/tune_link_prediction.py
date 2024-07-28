# experiments/tune_link_prediction.py

import logging
import os
import sys
from functools import partial
from typing import Any, Dict

import lightning as L
import psutil
import ray
import torch
import torch_geometric.transforms as T
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from torch_geometric.typing import Metadata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from halvesting_geometric.modules import LinkPrediction
from halvesting_geometric.utils import logging_config
from halvesting_geometric.utils.argparsers import TuneLinkPredictionArgparse
from halvesting_geometric.utils.data import LinkPredictionDataModule

GNN = "sage"
BATCH_SIZE = 128
NUM_NEIGHBORS = [32, 16]
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
NUM_NODES_MAP = {
    "all": {"paper": 18662037, "author": 238397, "affiliation": 96105, "domain": 20},
    "en": {"paper": 9395826, "author": 204804, "affiliation": 88867, "domain": 18},
    "fr": {"paper": 10686070, "author": 65124, "affiliation": 21636, "domain": 16},
}


logging_config()


def train_tune(
    config: Dict[str, Any],
    dataset: LinkPredictionDataModule,
    gnn: str,
    callback: TuneReportCheckpointCallback,
    metadata: Metadata,
    batch_size: int,
    author_num_nodes: int,
    institution_num_nodes: int,
    domain_num_nodes: int,
    accelerator: str,
):
    logging.info("Creating model...")
    model = LinkPrediction(
        gnn=gnn,  # type: ignore
        metadata=metadata,
        batch_size=batch_size,
        author_num_nodes=author_num_nodes,
        institution_num_nodes=institution_num_nodes,
        domain_num_nodes=domain_num_nodes,
        hidden_channels=config["hidden_channels"],
        dropout=config["dropout"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    logging.info("Training and testing model...")
    trainer = L.Trainer(
        accelerator=accelerator, devices=-1, max_epochs=2, callbacks=[callback]
    )
    trainer.fit(model=model, datamodule=dataset)


if __name__ == "__main__":
    args = TuneLinkPredictionArgparse.parse_known_args()

    try:
        torch.set_float32_matmul_precision("medium")
    except Exception as e:
        logging.error(f"Unable to activate TensorCore:\n{e}")

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

    dataset = LinkPredictionDataModule(
        data_dir=args.root_dir,
        batch_size=BATCH_SIZE,
        num_neighbors=NUM_NEIGHBORS,
        lang=args.lang_,
        neg_sampling_ratio=2.0,
        num_proc=num_proc,
        num_val=0.3,
        num_test=0.0,
        add_negative_train_samples=True,
        shuffle_train=False,
        shuffle_val=False,
        persistent_workers=False,
    )

    callback = TuneReportCheckpointCallback(
        {"loss": "val_loss", "roc_auc": "val_roc_auc"}, on="validation_end"
    )

    config = {
        "hidden_channels": tune.choice([16, 32, 64, 128]),
        "dropout": tune.uniform(0.1, 0.5),
        "lr": tune.choice([1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
        "weight_decay": tune.loguniform(1e-7, 1e-4),
    }
    ray.init(_temp_dir=f"{args.storage_path}/tmp")
    tune.run(
        partial(
            train_tune,
            gnn=GNN,
            callback=callback,
            dataset=dataset,
            accelerator=accelerator,
            metadata=METADATA,
            batch_size=BATCH_SIZE,
            author_num_nodes=NUM_NODES_MAP[args.lang_]["author"],
            institution_num_nodes=NUM_NODES_MAP[args.lang_]["affiliation"],
            domain_num_nodes=NUM_NODES_MAP[args.lang_]["domain"],
        ),
        metric="roc_auc",
        mode="max",
        name=f"Tune Link Prediction {GNN}",
        resources_per_trial={"cpu": num_proc, "gpu": 1},
        num_samples=33,
        search_alg=HyperOptSearch(),
        scheduler=AsyncHyperBandScheduler(),
        storage_path=args.storage_path,
        config=config,
        max_concurrent_trials=1,
    )
