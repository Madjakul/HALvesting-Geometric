# halph/models/bbp.py

import torch.nn as nn
from transformers import BigBirdPegasusModel
from transformers.tokenization_utils_base import BatchEncoding


class BigBirdPegasus(nn.Module):
    """Pass.

    Attributes
    ----------
    """

    def __init__(self):
        self.model = BigBirdPegasusModel.from_pretrained(
            "google/bigbird-pegasus-large-arxiv"
        )

    def forward(self, inputs: BatchEncoding):
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state[:, 0]
        return emb
