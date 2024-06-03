# halph/utils/data/link_prediction_metadata.py

import asyncio
import logging
import os
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List

import aiofiles
import fasteners
import pandas as pd
from datasets import DatasetDict
from tqdm import tqdm

from halph.utils import helpers


class LinkPredictionMetadata:
    """Pass.

    Atributes
    ---------

    Parameters
    ----------
    """

    def __init__(
        self, dataset: DatasetDict, root_dir: str, json_dir: str, xml_dir: str
    ):
        self.dataset = dataset
        self.root_dir = helpers.check_dir(root_dir)
        self.raw_dir = helpers.check_dir(osp.join(root_dir, "raw"))
        self.json_dir = json_dir
        self.xml_dir = xml_dir

    def __call__(self):
        pass

    @property
    def json_file_names(self):
        _json_file_names = os.listdir(self.json_dir)
        if not _json_file_names:
            return []
        json_file_names = [
            json_file_name
            for json_file_name in _json_file_names
            if json_file_name.endswith(".json")
        ]
        return json_file_names

    @property
    def xml_file_names(self):
        _json_file_names = os.listdir(self.json_dir)
        if not _json_file_names:
            return []
        json_file_names = [
            json_file_name
            for json_file_name in _json_file_names
            if json_file_name.endswith(".json")
        ]
        return json_file_names

    @fasteners.interprocess_lock("tmp/nodes.lock")
    def _node_worker(self):
        pass

    def compute_nodes(self, batch: Dict[str, List[str]]):
        pass

    def compute_edges(self, batch: Dict[str, List[str]]):
        pass
