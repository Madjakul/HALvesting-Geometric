# halph/models/bbp_link_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BigBirdPegasusModel, RobertaModel
from transformers.tokenization_utils_base import BatchEncoding


class LanguageModelLinkClassifier(nn.Module):
    """Pass.

    Parameters
    ----------

    Atributes
    ---------
    """

    def __init__(self, hidden_channels: int, dropout: float, device: str):
        super().__init__()
        self.lm = RobertaModel.from_pretrained("FacebookAI/roberta-base").to(device)
        hidden_size = self.lm.config.hidden_size
        self.dropout = dropout
        self.linear1 = nn.Linear(
            hidden_size + hidden_channels, hidden_size + hidden_channels
        )
        self.linear2 = nn.Linear(hidden_size + hidden_channels, hidden_channels)

    def forward(
        self,
        x_author: torch.Tensor,
        x_paper: torch.Tensor,
        edge_label_index: torch.Tensor,
        inputs: BatchEncoding,
    ):
        outputs = self.lm(
            # input_ids=inputs["input_ids"][0].unsqueeze(0),
            # attention_mask=inputs["attention_mask"][0].unsqueeze(0),
            **inputs
        )
        emb = outputs.last_hidden_state[:, 0]
        # for i in range(1, len(inputs["input_ids"])):
        #     outputs = self.bbp(
        #         input_ids=inputs["input_ids"][i].unsqueeze(0),
        #         attention_mask=inputs["attention_mask"][i].unsqueeze(0),
        #     )
        #     emb = torch.cat((emb, outputs.last_hidden_state[:, 0]), 0)

        # Convert node embeddings to edge-level representations:
        edge_feat_author = x_author[edge_label_index[0]]
        edge_feat_paper = x_paper[edge_label_index[1]]
        edge_feat_paper = torch.cat((emb, edge_feat_paper), 1)
        edge_feat_paper = self.linear1(edge_feat_paper)
        edge_feat_paper = F.gelu(edge_feat_paper)
        edge_feat_paper = F.dropout(edge_feat_paper, p=self.dropout)
        edge_feat_paper = self.linear2(edge_feat_paper)
        edge_feat_paper = F.gelu(edge_feat_paper)
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_author * edge_feat_paper).sum(dim=-1)
