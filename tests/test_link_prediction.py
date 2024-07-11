# tests/test_gcn.py

import pytest
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from halvesting_geometric.modules import LinkPrediction

# Constants for testing
HIDDEN_CHANNELS = 64
DROPOUT = 0.0
GNN_TYPES = ["sage", "gat", "rggc"]
RANDOM_SEED = 42

# Setting a random seed for reproducibility
torch.manual_seed(RANDOM_SEED)


# Function to create a mock dataset
def create_mock_dataset():
    data = HeteroData()

    # Author nodes
    data["author"].num_nodes = 10
    data["author"].n_id = torch.arange(10)

    # Paper nodes
    data["paper"].num_nodes = 15
    data["paper"].n_id = torch.arange(15)

    # Institution nodes
    data["institution"].num_nodes = 5
    data["institution"].n_id = torch.arange(5)

    # Domain nodes
    data["domain"].num_nodes = 3
    data["domain"].n_id = torch.arange(3)

    # Edges between author and paper
    data["author", "writes", "paper"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        dtype=torch.long,
    )
    data["author", "writes", "paper"].edge_label = torch.ones(10)

    data["author", "affiliated_with", "institution"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]],
        dtype=torch.long,
    )
    data["paper", "has_domain", "domain"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]],
        dtype=torch.long,
    )

    # Apply transformations
    transform = T.ToUndirected()
    data = transform(data)

    return data


# Pytest fixtures
@pytest.fixture
def mock_data():
    return create_mock_dataset()


@pytest.fixture
def mock_dataloader(mock_data):
    transform = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=("author", "writes", "paper"),
        rev_edge_types=("paper", "rev_writes", "author"),
    )
    data, _, _ = transform(mock_data)

    edge_label_index = data["author", "writes", "paper"].edge_label_index
    edge_label = data["author", "writes", "paper"].edge_label
    dataloader = LinkNeighborLoader(
        data=data,
        num_neighbors=[4, 4],
        neg_sampling_ratio=2.0,
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label=edge_label,
        batch_size=4,
        shuffle=False,
    )

    return dataloader


@pytest.fixture
def metadata(mock_data):
    return mock_data.metadata()


@pytest.mark.parametrize("gnn_type", GNN_TYPES)
def test_forward_pass(mock_data, mock_dataloader, metadata, gnn_type):
    mock_batch = next(iter(mock_dataloader))

    model = LinkPrediction(
        gnn=gnn_type,
        metadata=metadata,
        paper_num_nodes=mock_data["paper"].num_nodes,
        author_num_nodes=mock_data["author"].num_nodes,
        institution_num_nodes=mock_data[
            "institution"
        ].num_nodes,  # Assuming no institution nodes in mock dataset
        domain_num_nodes=mock_data[
            "domain"
        ].num_nodes,  # Assuming no domain nodes in mock dataset
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    )

    # Forward pass
    output = model(mock_batch)

    # Check output type and shape
    assert isinstance(output, torch.Tensor)
    assert output.shape == (
        mock_batch["author", "writes", "paper"].edge_label_index.shape[1],
    )

    # Check if the output is deterministic by running the forward pass again and comparing
    torch.manual_seed(RANDOM_SEED)
    output_again = model(mock_batch)
    assert torch.allclose(output, output_again), "Outputs are not deterministic"
