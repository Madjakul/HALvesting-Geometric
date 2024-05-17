# halph/utils/data/metadata.py

import logging
import multiprocessing
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from tqdm import tqdm

from halph.utils import helpers


class Metadata(ABC):
    """Abstract class.

    Parameters
    ----------

    Attributes
    ----------
    """

    headers: Dict[str, Dict[str, Any]]

    def __init__(self, template: str, json_dir_path: str, xml_dir_path: str):
        self.template = template
        self.xml_dir_path = xml_dir_path
        self.xml_files = os.listdir(xml_dir_path)
        self.xml_file_paths = [
            os.path.join(xml_dir_path, xml_file) for xml_file in self.xml_files
        ]
        json_files = os.listdir(json_dir_path)
        json_file_paths = [
            os.path.join(json_dir_path, json_file_path) for json_file_path in json_files
        ]
        headers = helpers.jsons_to_dict(json_file_paths, on="halid")
        logging.info("Filtering the headers...")
        self.headers = self._filter_headers(headers)

    def _filter_headers(self, headers: Dict[str, Dict[str, Any]]):
        keys = list(headers.keys())
        for key in tqdm(keys):
            if f"{key}.grobid.tei.xml" in self.xml_files:
                continue
            del headers[key]
        return headers

    @abstractmethod
    def _worker(
        self,
        q: multiprocessing.Queue,
        header: Dict[str, Dict[str, Any]],
        path,
        str,
    ):
        raise NotImplementedError

    @abstractmethod
    def _listener(self, q: multiprocessing.Queue, output_dir_path: str):
        raise NotImplementedError

    @abstractmethod
    def build(self, output_dir_path: str):
        raise NotImplementedError
