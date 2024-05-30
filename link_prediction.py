# link_prediction.py

import logging

import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader

import wandb
from halph.benchmarks import BenchMarkLinkPrediction
from halph.models import LinkPrediction
from halph.trainers import LinkPredictionTrainer
from halph.utils import logging_config
from halph.utils.argparsers import LinkPredictionArgparse
from halph.utils.data import LinkPredictionDataset

logging_config()

GNNS = ("sage", "gat", "rggc")
WEIGHT_DECAY = 0
LR = 5e-3
HIDDEN_CHANNELS = 64
DROPOUT = 0.1


def main(gnn: str, run: int):
    wandb.init(project="HALph", entity="madjakul", name=f"link-prediction-{gnn}-{run}")

    dataset = LinkPredictionDataset("./data/mock")
    data = dataset[0]
    data = T.ToUndirected()(data)
    # data = T.AddSelfLoops()(data)

    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=("author", "writes", "paper"),
        rev_edge_types=("paper", "rev_writes", "author"),
    )
    train_data, val_data, test_data = transform(data)

    edge_label_index = train_data["author", "writes", "paper"].edge_label_index
    edge_label = train_data["author", "writes", "paper"].edge_label
    train_dataloader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[128, 16],
        neg_sampling_ratio=2.0,
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label=edge_label,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    edge_label_index = val_data["author", "writes", "paper"].edge_label_index
    edge_label = val_data["author", "writes", "paper"].edge_label
    val_dataloader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[128, 16],
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label=edge_label,
        batch_size=4 * 4,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    model = LinkPrediction(
        gnn=gnn,
        metadata=data.metadata(),
        paper_num_nodes=data["paper"].num_nodes,
        author_num_nodes=data["author"].num_nodes,
        institution_num_nodes=data["institution"].num_nodes,
        domain_num_nodes=data["domain"].num_nodes,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    )

    trainer = LinkPredictionTrainer(
        model=model, lr=LR, device="cpu", weight_decay=WEIGHT_DECAY
    )
    trainer.train(train_dataloader, val_dataloader, epochs=50)

    edge_label_index = test_data["author", "writes", "paper"].edge_label_index
    edge_label = test_data["author", "writes", "paper"].edge_label
    test_dataloader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[128, 16],
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label=edge_label,
        batch_size=4 * 4,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    auc = BenchMarkLinkPrediction.run(model, test_dataloader, "cpu")
    wandb.log({"test-auc": auc})
    logging.info(f"test-auc: {auc}")
    # wandb.teardown()
    wandb.finish()


if __name__ == "__main__":
    args = LinkPredictionArgparse.parse_known_args()
    if args.gnn == "gat":
        WEIGHT_DECAY = 5e-4
        DROPOUT = 0.5
    if args.gnn == "rggc":
        WEIGHT_DECAY = 5e-4
        HIDDEN_CHANNELS = 16
        LR = 1e-2
    main(args.gnn, args.run)
