# halves_geometric/utils/data/link_prediction_datamodule.py

import logging
from typing import List

import lightning as L
import torch_geometric.transforms as T
from filelock import FileLock
from torch_geometric.loader import LinkNeighborLoader

from halvesting_geometric.utils.data.link_prediction_dataset import \
    LinkPredictionDataset


class LinkPredictionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_neighbors: List[int],
        lang: str,
        neg_sampling_ratio: float,
        num_proc: int,
        num_val: float,
        num_test: float,
        add_negative_train_samples: bool,
        shuffle_train: bool,
        shuffle_val: bool,
        persistent_workers: bool,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.lang = lang
        self.num_proc = num_proc
        self.neg_sampling_ratio = neg_sampling_ratio
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.persistent_workers = persistent_workers
        self.transform = T.Compose(
            [
                T.ToUndirected(),
                T.RandomLinkSplit(
                    num_val=num_val,
                    num_test=num_test,
                    neg_sampling_ratio=neg_sampling_ratio,
                    add_negative_train_samples=add_negative_train_samples,
                    edge_types=("author", "writes", "paper"),
                    rev_edge_types=("paper", "rev_writes", "author"),
                ),
            ]
        )

    def prepare_data(self):
        LinkPredictionDataset(self.data_dir, lang=self.lang, load=False)

    def setup(self, stage=None):
        with FileLock(f"{self.data_dir}.lock"):
            dataset = LinkPredictionDataset(self.data_dir, lang=self.lang)
            self.data = dataset[0]
            self.train_data, self.val_data, self.test_data = self.transform(self.data)

    def train_dataloader(self):
        edge_label_index = self.train_data["author", "writes", "paper"].edge_label_index
        edge_label = self.train_data["author", "writes", "paper"].edge_label
        train_dataloader = LinkNeighborLoader(
            data=self.train_data,
            num_neighbors=self.num_neighbors,
            neg_sampling_ratio=self.neg_sampling_ratio,
            edge_label_index=(("author", "writes", "paper"), edge_label_index),
            edge_label=edge_label,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_proc,
            persistent_workers=self.persistent_workers,
        )
        return train_dataloader

    def val_dataloader(self):
        edge_label_index = self.val_data["author", "writes", "paper"].edge_label_index
        edge_label = self.val_data["author", "writes", "paper"].edge_label
        val_dataloader = LinkNeighborLoader(
            data=self.val_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=(("author", "writes", "paper"), edge_label_index),
            edge_label=edge_label,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_proc,
            persistent_workers=self.persistent_workers,
        )
        return val_dataloader

    def test_dataloader(self):
        edge_label_index = self.test_data["author", "writes", "paper"].edge_label_index
        edge_label = self.test_data["author", "writes", "paper"].edge_label
        test_dataloader = LinkNeighborLoader(
            data=self.test_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=(("author", "writes", "paper"), edge_label_index),
            edge_label=edge_label,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_proc,
            persistent_workers=self.persistent_workers,
        )
        return test_dataloader
