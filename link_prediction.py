# link_prediction.py

import logging

import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader

from halph.benchmarks import BenchMarkLinkPrediction
from halph.models import LinkPrediction
from halph.trainers import LinkPredictionTrainer
from halph.utils import logging_config
from halph.utils.data import LinkPredictionDataset

logging_config()


if __name__ == "__main__":
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
        num_neighbors=[64, 28],
        neg_sampling_ratio=2.0,
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label=edge_label,
        batch_size=4,
        shuffle=True,
    )
    edge_label_index = val_data["author", "writes", "paper"].edge_label_index
    edge_label = val_data["author", "writes", "paper"].edge_label
    val_dataloader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[64, 28],
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label=edge_label,
        batch_size=4 * 4,
        shuffle=False,
    )

    model = LinkPrediction(
        gnn="graph_sage",
        metadata=data.metadata(),
        paper_num_nodes=data["paper"].num_nodes,
        author_num_nodes=data["author"].num_nodes,
        institution_num_nodes=data["institution"].num_nodes,
        domain_num_nodes=data["domain"].num_nodes,
        paper_num_features=data["paper"].num_features,
        hidden_channels=64,
        dropout=0.1,
    )

    trainer = LinkPredictionTrainer(model=model, lr=0.001, device="cpu", weight_decay=0)
    trainer.train(train_dataloader, val_dataloader, epochs=50)

    edge_label_index = test_data["author", "writes", "paper"].edge_label_index
    edge_label = test_data["author", "writes", "paper"].edge_label
    test_dataloader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[64, 28],
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label=edge_label,
        batch_size=4 * 4,
        shuffle=False,
    )

    auc = BenchMarkLinkPrediction.run(model, test_dataloader, "cpu")
    logging.info(f"test AUC: {auc}")
