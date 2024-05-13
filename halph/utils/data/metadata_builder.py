# halph/utils/data/metadata_builder.py

import asyncio
import json
import logging
import os
from typing import Any, Dict, Literal

import aiofiles
from datasets import DatasetDict
from lxml import etree
from tqdm import tqdm

from halph.utils import helpers


class MetadataBuilder:
    """This class builds the raw documents used to creates graphs out of HAL
    that can be loaded by **torch_geometric**.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------
    """

    headers: Dict[str, Dict[str, Any]]

    def __init__(self, template: str, dataset: DatasetDict, json_dir_path: str):
        self.template = template
        self.dataset = dataset
        json_files = os.listdir(json_dir_path)
        json_file_paths = [
            os.path.join(json_dir_path, json_file_path) for json_file_path in json_files
        ]
        headers = helpers.read_jsons(json_file_paths)

    def __call__(self, output_dir_path: str):
        pass

    def _compute_nodes(self)

    def _build(self):
        pass

    def _build_enriched(self):
        pass

    def _build_from_mag(self):
        pass

    def build(self, output_dir_path: str):
        pass
