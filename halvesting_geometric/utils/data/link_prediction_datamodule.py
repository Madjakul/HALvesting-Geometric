# halves_geometric/utils/data/link_prediction_datamodule.py

from typing import List

import lightning as L
import torch_geometric.transforms as T
from filelock import FileLock
from torch_geometric.loader import LinkNeighborLoader

from halvesting_geometric.utils.data.link_prediction_dataset import (
    LinkPredictionDataset,
)


class LinkPredictionDataModule(L.LightningDataModule):
    """DataModule for link prediction tasks. The dataset is split into train,
    validation, and test sets.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the dataset is stored.
    batch_size : int
        Number of samples in a batch.
    num_neighbors : List[int]
        Number of neighbors to sample for each layer.
    lang : str
        Language of the dataset.
    neg_sampling_ratio : float
        Ratio of negative samples to positive samples.
    num_proc : int
        Number of processes to use for data loading.
    num_val : float
        Ratio of validation samples.
    num_test : float
        Ratio of test samples.
    add_negative_train_samples : bool
        Whether to add negative samples to the training set.
    shuffle_train : bool
        Whether to shuffle the training set.
    shuffle_val : bool
        Whether to shuffle the validation set.
    persistent_workers : bool
        Whether to keep the workers alive between epochs.
    
    Attributes
    ----------
    data_dir : str
        Path to the directory where the dataset is stored.
    batch_size : int
        Number of samples in a batch.
    num_neighbors : List[int]
        Number of neighbors to sample for each layer.
    lang : str
        Language of the dataset.
    num_proc : int
        Number of processes to use for data loading.
    neg_sampling_ratio : float
        Ratio of negative samples to positive samples.
    shuffle_train : bool
        Whether to shuffle the training set.
    shuffle_val : bool
        Whether to shuffle the validation set.
    persistent_workers : bool
        Whether to keep the workers alive between epochs.
    transform : torch_geometric.transforms.Compose
        Data transformation pipeline.
    data : torch_geometric.data.Data
        The dataset.
    train_data : torch_geometric.data.Data
        The training set.
    val_data : torch_geometric.data.Data
        The validation set.
    test_data : torch_geometric.data.Data
        The test set.
    
    Examples
    --------
    >>> from halvesting_geometric.utils.data.link_prediction_datamodule import \
    ...     LinkPredictionDataModule
    >>> datamodule = LinkPredictionDataModule(
    ...     data_dir="data",
    ...     batch_size=64,
    ...     num_neighbors=[10, 10],
    ...     lang="en",
    ...     neg_sampling_ratio=1,
    ...     num_proc=4,
    ...     num_val=0.1,
    ...     num_test=0.1,
    ...     add_negative_train_samples=True,
    ...     shuffle_train=True,
    ...     shuffle_val=True,
    ...     persistent_workers=True,
    ... )
    >>> datamodule.prepare_data()
    >>> datamodule.setup()
    """

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
