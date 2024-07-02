# halvesting_geometric/models/modeling_graph_sage.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSage(nn.Module):
    """GraphSage model.

    Parameters
    ----------
    in_channels: int
        Number of input features.
    hidden_channels: int
        Number of hidden features.
    out_channels: int
        Number of output features.
    dropout: float
        Dropout probability.

    Attributes
    ----------
    dropout: float
        Dropout probability.
    conv1: SAGEConv
        First GraphSage convolution layer.
    conv2: SAGEConv
        Second GraphSage convolution layer.

    Examples
    --------
    >>> import torch
    >>> from halvesting_geometric.models import GraphSage
    >>> model = GraphSage(128, 128, 128, 0.5)
    >>> input = torch.randn(10, 128)
    >>> edge_index = torch.randint(0, 10, (2, 100))
    >>> output = model(input, edge_index)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
        **kwargs
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Graph edge indices.
        edge_weight : Optional[torch.Tensor], optional
            Edge weights, by default None.

        Returns
        -------
        x : torch.Tensor
            Node embeddings.
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.conv2(x, edge_index)
        return x
