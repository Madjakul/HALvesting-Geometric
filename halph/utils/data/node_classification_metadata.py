# halph/utils/data/node_classification_metadata.py

import logging
import multiprocessing
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List

from datasets import DatasetDict
from lxml import etree
from lxml.etree import ElementTree
from tqdm import tqdm

from halph.utils import helpers
from halph.utils.data.metadata import Metadata


class NodeClassificationMetadata(Metadata):
    """This class builds the raw documents used to creates graphs out of HAL
    that can be loaded by **torch_geometric**.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------
    """

    def __init__(
        self,
        dataset: DatasetDict,
        json_dir_path: str,
        xml_dir_path: str,
        num_proc: int,
    ):
        super().__init__(
            template="node_classification",
            json_dir_path=json_dir_path,
            xml_dir_path=xml_dir_path,
        )
        self.num_proc = num_proc
        self.dataset = dataset.remove_columns("text")
        field_of_study = dataset["domain"]
        # Nodes
        self.fields_of_study = list(
            set([field for fields in field_of_study for field in fields])
        )
        self.authors = OrderedDict()
        self.papers = []
        self.institutions = []

    def __call__(self, output_dir_path: str):
        self.build(output_dir_path)

    def _get_citations(self, root: ElementTree):
        bibl_structs = root.xpath("//text/back/div/listBibl/biblStruct")
        c_titles = []
        for bibl_struct in bibl_structs:
            c_titles.append(self._get_citations_title(bibl_struct))
        return c_titles

    def _add_paper_node(self, title: str):
        if title not in self.papers:
            self.papers.append(title)

    def _add_institution_node(self, authors: List[Dict[str, Any]]):
        for author in authors:
            institutions = author["affiliations"]
            for institution in institutions:
                if institution not in self.institutions:
                    self.institutions.append(institution)

    def _add_author_nodes(self, authors: List[Dict[str, Any]]):
        for author in authors:
            halauthorid = author["halauthorid"]
            if halauthorid == "0":
                continue
            name = author["name"]
            if halauthorid in self.authors and self.authors[halauthorid] != name:
                self.authors[halauthorid].append(name)
            elif halauthorid not in self.authors:
                self.authors[halauthorid] = [name]

    def _get_edges(
        self,
        local_authors: List[Dict[str, Any]],
        title: str,
        c_titles: List[str],
        domains: List[str],
    ):
        edges = defaultdict(list)
        paper_index = self.papers.index(title)
        for c_title in c_titles:
            edges["paper__cites__paper"].append(
                (paper_index, self.papers.index(c_title))
            )
        for domain in domains:
            edges["paper__has_topic__field_of_study"].append(
                (paper_index, self.fields_of_study.index(domain))
            )
        for author in local_authors:
            halauthorid = author["halauthorid"]
            if halauthorid == "0":
                continue
            author_index = list(self.authors).index(halauthorid)
            local_institutions = author["affiliations"]
            for institution in local_institutions:
                edges["author__affiliated_with__institution"].append(
                    (author_index, self.institutions.index(institution))
                )
            edges["author__writes__paper"].append((author_index, paper_index))
        return edges

    def _worker(self, q: multiprocessing.Queue, document: Dict[str, Any], path: str):
        # Get correct header
        halid = document["halid"]
        domains = document["domain"]
        header = self.headers[halid]

        # Process XML file with the header.
        with open(path, "r", encoding="utf-8") as xmlf:
            xml = xmlf.read()
        root = etree.fromstring(xml.encode("utf-8"))
        for elem in root.getiterator():
            # Skip comments and processing instructions,
            # because they do not have names
            if not (
                isinstance(elem, etree._Comment)
                or isinstance(elem, etree._ProcessingInstruction)
            ):
                # Remove a namespace URI in the element's name
                elem.tag = etree.QName(elem).localname
        # Remove unused namespace declarations
        etree.cleanup_namespaces(root)

        # Get nodes
        title = header["title"]
        self._add_paper_node(title)
        c_titles = self._get_citations(root)
        for c_title in c_titles:
            self._add_paper_node(c_title)
        local_authors = header["authors"]
        self._add_author_nodes(local_authors)
        self._add_institution_node(local_authors)

        # Get edges
        edges = self._get_edges(
            local_authors=local_authors, title=title, c_titles=c_titles, domains=domains
        )

        q.put(edges)

    def _listener(self, q: multiprocessing.Queue, output_dir_path: str):
        edges_dir_path = helpers.check_dir(os.path.join(output_dir_path, "edges"))
        while True:
            edges = q.get()
            if q is None:
                break
            for k, v in edges.items():
                edges_file_path = os.path.join(edges_dir_path, f"{k}.csv")
                with open(edges_file_path, "a", encoding="utf-8") as csvf:
                    for edge in v:
                        csvf.write(f"{edge[0]},{edge[1]}\n")
                        csvf.flush()

    def build(self, output_dir_path: str):
        manager = multiprocessing.Manager()
        q = manager.Queue()
        file_pool = multiprocessing.Pool(1)
        file_pool.apply_async(self._listener, (q, output_dir_path))

        pool = multiprocessing.Pool(self.num_proc - 1)
        jobs = []
        for document in tqdm(self.dataset):
            halid = document["halid"]
            path = os.path.join(self.xml_dir_path, f"{halid}.grobid.tei.xml")
            if path not in self.xml_file_paths:
                continue
            job = pool.apply_async(self._worker, (q, document, path))
            jobs.append(job)

        for job in jobs:
            job.get()

        q.put(None)
        pool.close()
        pool.join()

    @staticmethod
    def _get_citations_title(bibl_struct: etree._Element):
        title_element = bibl_struct.find(".//title[@type='main']")
        title = title_element.text if title_element is not None else ""
        return title
