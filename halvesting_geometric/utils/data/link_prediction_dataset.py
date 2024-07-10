# halvesting_geometric/utils/data/link_prediction_dataset.py

import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import (HeteroData, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.utils import coalesce


class LinkPredictionDataset(InMemoryDataset):
    """Base class for link prediction datasets.

    Parameters
    ----------
    root : str
        Directory where the dataset should be stored.
    preprocess : str, optional
        Preprocessing method to apply to the dataset.
    transform : Callable, optional
        A function/transform that takes in an HeteroData object and returns a
        transformed version.
    pre_transform : Callable, optional
        A function/transform that takes in an HeteroData object and returns a
        transformed version.
    force_reload : bool, optional
        If set to :obj:`True`, the dataset will be re-downloaded and preprocessed, even
        if it already exists on disk.
    lang : str, optional
        Language of the dataset. Can be :obj:`"en"`, :obj:`"fr"` or :obj:`"all"`.

    Attributes
    ----------
    url : str
        URL to download the dataset.
    preprocess : str
        Preprocessing method to apply to the dataset.

    Examples
    --------
    >>> from halvesting_geometric.utils.data import LinkPredictionDataset
    >>> dataset = LinkPredictionDataset("./data")
    Downloading https://huggingface.co/datasets/Madjakul/HALvest-Geometric/resolve/main/raw.zip
    Extracting data/raw.zip
    Processing...
    Done!
    >>> data = dataset[0]
    >>> data
    HeteroData(
    paper={
        num_features=0,
        num_nodes=18662037,
        index=[18662037],
    },
    author={
        index=[238397],
        num_nodes=238397,
        num_features=0,
    },
    affiliation={
        index=[96105],
        num_nodes=96105,
        num_features=0,
    },
    domain={
        index=[20],
        num_nodes=20,
        num_features=0,
    },
    (author, writes, paper)={ edge_index=[2, 834644] },
    (author, affiliated_with, affiliation)={ edge_index=[2, 426030] },
    (paper, cites, paper)={ edge_index=[2, 22363817] },
    (paper, has_topic, domain)={ edge_index=[2, 136700] }
    )
    """

    url: str
    urls = {
        "en": "https://huggingface.co/datasets/Madjakul/HALvest-Geometric/resolve/main/raw-en.zip",
        "fr": "https://huggingface.co/datasets/Madjakul/HALvest-Geometric/resolve/main/raw-fr.zip",
        "all": "https://huggingface.co/datasets/Madjakul/HALvest-Geometric/resolve/main/raw.zip",
    }

    def __init__(
        self,
        root: str,
        preprocess: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        lang: Optional[str] = None,
    ):
        self.lang = lang
        if lang is None:
            self.url = self.urls["all"]
        else:
            assert lang in ["en", "fr", "all"]
            self.url = self.urls[lang]
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    def __repr__(self) -> str:
        return f"HALvest-Geometric-{self.lang}()"

    @property
    def raw_dir(self) -> str:
        if self.lang is None:
            return osp.join(self.root, "raw")
        return osp.join(self.root, f"raw-{self.lang}")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        file_names = ["nodes", "edges"]

        if self.preprocess is not None:
            file_names += [f"halph_{self.preprocess}_emb.pt"]

        return file_names

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        # Get paper nodes
        path = osp.join(self.raw_dir, "nodes", "papers.csv")
        df = pd.read_csv(
            path,
            sep="\t",
            dtype={
                "halid": str,
                "year": "Int64",
                "title": str,
                "lang": str,
                "domain": str,
                "paper_idx": "Int64",
            },
            chunksize=1000000,
        )
        data["paper"].num_features = 0
        data["paper"].num_nodes = 0
        index = np.empty((0,), dtype=np.int64)
        for chunk in df:
            index = np.append(index, chunk.index.values)
            data["paper"].num_nodes += chunk.shape[0]
        data["paper"].index = torch.from_numpy(index)

        # Get author nodes
        path = osp.join(self.raw_dir, "nodes", "authors.csv")
        df = pd.read_csv(
            path,
            sep="\t",
            dtype={"name": str, "halauthorid": "Int64", "author_idx": "Int64"},
        )
        data["author"].index = torch.from_numpy(df.index.values)
        data["author"].num_nodes = df.shape[0]
        data["author"].num_features = 0

        # Get affiliation nodes
        path = osp.join(self.raw_dir, "nodes", "affiliations.csv")
        df = pd.read_csv(
            path,
            sep="\t",
            dtype={"affiliations": str, "affiliation_idx": "Int64"},
        )
        data["affiliation"].index = torch.from_numpy(df.index.values)
        data["affiliation"].num_nodes = df.shape[0]
        data["affiliation"].num_features = 0

        # Get domain nodes
        path = osp.join(self.raw_dir, "nodes", "domains.csv")
        df = pd.read_csv(
            path,
            sep="\t",
            dtype={"domain": str, "domain_idx": "Int64"},
        )
        data["domain"].index = torch.from_numpy(df.index.values)
        data["domain"].num_nodes = df.shape[0]
        data["domain"].num_features = 0

        edge_dir_path = osp.join(self.raw_dir, "edges")

        # Get author <-> paper edges
        path = osp.join(edge_dir_path, "author__writes__paper.csv")
        df = pd.read_csv(
            path,
            sep="\t",
            dtype={"author_idx": "Int64", "paper_idx": "Int64"},
        )
        df = torch.from_numpy(df.values.astype(np.int64))
        df = df.t().contiguous()
        M, N = int(df[0].max() + 1), int(df[1].max() + 1)
        df = coalesce(df, num_nodes=max(M, N))
        data["author", "writes", "paper"].edge_index = df

        # Get author <-> affiliation edges
        path = osp.join(edge_dir_path, "author__affiliated_with__affiliation.csv")
        df = pd.read_csv(
            path,
            sep="\t",
            dtype={"author_idx": "Int64", "affiliation_idx": "Int64"},
        )
        df = torch.from_numpy(df.values.astype(np.int64))
        df = df.t().contiguous()
        M, N = int(df[0].max() + 1), int(df[1].max() + 1)
        df = coalesce(df, num_nodes=max(M, N))
        data["author", "affiliated_with", "affiliation"].edge_index = df

        # Get paper <-> paper edges
        path = osp.join(edge_dir_path, "paper__cites__paper.csv")
        df = pd.read_csv(
            path,
            sep="\t",
            dtype={"paper_idx": "Int64", "c_paper_idx": "Int64"},
            chunksize=1000000,
        )
        values = np.empty((0, 2), dtype=np.int64)
        for chunk in df:
            values = np.vstack([values, chunk.values])
        df = torch.from_numpy(values.astype(np.int64))
        df = df.t().contiguous()
        M, N = int(df[0].max() + 1), int(df[1].max() + 1)
        df = coalesce(df, num_nodes=max(M, N))
        data["paper", "cites", "paper"].edge_index = df

        # Get paper <-> domain edges
        path = osp.join(edge_dir_path, "paper__has_topic__domain.csv")
        df = pd.read_csv(
            path,
            sep="\t",
            dtype={"paper_idx": "Int64", "domain_idx": "Int64"},
        )
        df = torch.from_numpy(df.values.astype(np.int64))
        df = df.t().contiguous()
        M, N = int(df[0].max() + 1), int(df[1].max() + 1)
        df = coalesce(df, num_nodes=max(M, N))
        data["paper", "has_topic", "domain"].edge_index = df

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
