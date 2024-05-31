# halph/utils/data/bbp_dataset.py

import logging
import os.path as osp
from typing import List

import pandas as pd
import torch
from datasets import DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer


class BigBirdPegasusDataset:
    """Pass.

    Parameters
    ----------

    Attributes
    ----------
    """

    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

    def __init__(self, root: str, dataset: DatasetDict, max_length: int):
        self.dataset = dataset
        self.max_length = max_length
        path = osp.join(root, "raw", "nodes", "id_paper.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            names=["id", "halid", "year", "name"],
            index_col=0,
            dtype={"id": int, "halid": str, "year": int, "name": str},
        )
        logging.info("Processing paper nodes...")
        self.n_id_to_halid = {}
        for idx, row in df.iterrows():
            self.n_id_to_halid[idx] = row["halid"]
        self.halid_to_idx = {}
        logging.info("processing text data...")
        for idx, data in enumerate(tqdm(dataset)):
            self.halid_to_idx[data["halid"]] = idx

    def _tokenize(self, batch_text: List[str]):
        inputs = self.tokenizer(
            batch_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def get(self, batch: torch.Tensor):
        batch_text = []
        for n_id in batch.squeeze(0):
            halid = self.n_id_to_halid[n_id.item()]
            if halid == "0":
                batch_text.append("")
                continue
            dataset_idx = self.halid_to_idx[halid]
            batch_text.append(self.dataset[dataset_idx]["text"])
        return self._tokenize(batch_text)
