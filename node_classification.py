# node_classification.py

import logging

import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader

from halph.models import NodeClassification
from halph.trainers import NodeClassificationTrainer
from halph.utils import logging_config
from halph.utils.data import NodeClassificationDataset

logging_config()


if __name__ == "__main__":
    dataset = NodeClassificationDataset("./data/mock")
    data = dataset[0]
    data = T.ToUndirected()(data)
    # data = T.AddSelfLoops()(data)

    split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    split_data = split(data)

    train_dataloader = NeighborLoader(
        split_data,
        batch_size=4,
        input_nodes=("paper", split_data["paper"].train_mask),
        num_neighbors=[512] * 2,
        shuffle=True,
    )
    val_dataloader = NeighborLoader(
        split_data,
        batch_size=4,
        input_nodes=("paper", split_data["paper"].val_mask),
        num_neighbors=[512] * 2,
        shuffle=True,
    )
    test_dataloader = NeighborLoader(
        split_data,
        batch_size=4,
        input_nodes=("paper", split_data["paper"].test_mask),
        num_neighbors=[512] * 2,
    )

    model = NodeClassification(
        gnn="graph_sage",
        metadata=split_data.metadata(),
        paper_num_nodes=split_data["paper"].num_nodes,
        author_num_nodes=split_data["author"].num_nodes,
        institution_num_nodes=split_data["institution"].num_nodes,
        paper_num_features=split_data["paper"].num_features,
        hidden_channels=16,
        num_classes=split_data["paper"].num_classes,
        dropout=0.5,
    )
    # model = to_hetero(model, metadata=split_data.metadata())

    trainer = NodeClassificationTrainer(
        model=model, lr=0.01, device="cpu", weight_decay=5e-4
    )
    trainer.train(train_dataloader, val_dataloader, epochs=50)
