# halph/utils/data/metadata.py

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Dict

from datasets import DatasetDict
from lxml import etree
from lxml.etree import ElementTree
from tqdm import tqdm

from halph.utils import helpers

_DOMAINS = (
    "shs",
    "info",
    "sdv",
    "spi",
    "phys",
    "math",
    "chim",
    "sde",
    "sdu",
    "stat",
    "scco",
    "qfin",
)


class Metadata(ABC):
    """Abstract class.

    Parameters
    ----------

    Attributes
    ----------
    """

    headers: Dict[str, Dict[str, Any]]

    def __init__(
        self, template: str, json_dir_path: str, xml_dir_path: str, dataset: DatasetDict
    ):
        self.domains = _DOMAINS
        self.template = template
        self.dataset = dataset.remove_columns("text")
        # Nodes
        self.authors = OrderedDict()
        self.papers = OrderedDict()
        self.paper_domain = defaultdict(list)
        self.institutions = []

        self.xml_dir_path = xml_dir_path
        self.xml_files = os.listdir(xml_dir_path)
        logging.info(f"Found {len(self.xml_files)} XML files at {xml_dir_path}.")
        self.xml_file_paths = [
            os.path.join(xml_dir_path, xml_file) for xml_file in self.xml_files
        ]
        json_files = os.listdir(json_dir_path)
        logging.info(f"Found {len(json_files)} JSON files at {json_dir_path}.")
        json_file_paths = [
            os.path.join(json_dir_path, json_file_path) for json_file_path in json_files
        ]
        headers = helpers.jsons_to_dict(json_file_paths, on="halid")
        logging.info("Filtering the headers...")
        self.headers = self._filter_headers(headers)

    @abstractmethod
    async def _build(self, q: asyncio.Queue):
        raise NotImplementedError

    @abstractmethod
    async def _process(self, document: Dict[str, Any], path: str):
        raise NotImplementedError

    @abstractmethod
    async def _save_edges(self, q: asyncio.Queue, output_dir_path: str):
        raise NotImplementedError

    def _filter_headers(self, headers: Dict[str, Dict[str, Any]]):
        keys = list(headers.keys())
        for key in tqdm(keys):
            if f"{key}.grobid.tei.xml" in self.xml_files:
                continue
            del headers[key]
        return headers

    def _get_citations(self, root: ElementTree):
        bibl_structs = root.xpath("//text/back/div/listBibl/biblStruct")
        c_titles = []
        c_years = []
        for bibl_struct in bibl_structs:
            c_title = self._get_citations_title(bibl_struct)
            c_year = self._get_citation_year(bibl_struct)
            if c_title == "":
                continue
            c_titles.append(c_title)
            c_years.append(c_year)
        return c_titles, c_years

    @abstractmethod
    def save_nodes(self, output_dir_path: str):
        raise NotImplementedError

    @abstractmethod
    def build(self, output_dir_path: str):
        raise NotImplementedError

    @staticmethod
    def _get_citations_title(bibl_struct: etree._Element):
        title_element = bibl_struct.find(".//title[@type='main']")
        title = title_element.text if title_element is not None else ""
        return title

    @staticmethod
    def _get_citation_year(bibl_struct: etree._Element):
        date_element = bibl_struct.find(".//imprint/date[@type='published']")
        year = (
            date_element.attrib["when"].split("-")[0]
            if date_element is not None
            else "0"
        )
        return year
