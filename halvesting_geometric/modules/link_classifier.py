# halvesting_geometric/models/link_classifier.py

import torch
import torch.nn as nn


class LinkClassifier(nn.Module):
    """Link classifier model."""

    def forward(
        self,
        x_author: torch.Tensor,
        x_paper: torch.Tensor,
        edge_label_index: torch.Tensor,
    ):
        """Forward pass.

        Parameters
        ----------
        x_author : torch.Tensor
            Author embeddings.
        x_paper : torch.Tensor
            Paper embeddings.
        edge_label_index : torch.Tensor
            Edge label indices.

        Returns
        -------
        torch.Tensor
            Predictions.
        """
        # Convert node embeddings to edge-level representations:
        edge_feat_author = x_author[edge_label_index[0]]
        edge_feat_paper = x_paper[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_author * edge_feat_paper).sum(dim=-1)
