# halph/utils/data/node_classification_metadata.py

import asyncio
import gzip
import logging
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List

import aiofiles
from datasets import DatasetDict
from tqdm import tqdm

from halph.utils import helpers
from halph.utils.data.metadata import Metadata

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

    def __init__(self, dataset: DatasetDict, json_dir_path: str, xml_dir_path: str):
        super().__init__(
            template="node_classification",
            json_dir_path=json_dir_path,
            xml_dir_path=xml_dir_path,
        )
        self.dataset = dataset.remove_columns("text")
        # Nodes
        self.authors = OrderedDict()
        self.papers = OrderedDict()
        self.institutions = []
        # Labels
        self.paper_domain = OrderedDict()

    def __call__(self, output_dir_path: str):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.build(output_dir_path))

    async def _build(self, q: asyncio.Queue):
        logging.info("Processing graph/text documents paired to headers...")
        for document in tqdm(self.dataset):
            halid = document["halid"]
            xml_file_path = os.path.join(self.xml_dir_path, f"{halid}.grobid.tei.xml")
            if xml_file_path not in self.xml_file_paths:
                continue
            edges = await self._process(document, xml_file_path)
            await q.put(edges)
        await q.put(None)

    async def _save_edges(self, q: asyncio.Queue, output_dir_path: str):
        edges_dir_path = helpers.check_dir(os.path.join(output_dir_path, "edges"))
        while True:
            edges = await q.get()
            if edges is None:
                break
            for k, v in edges.items():
                edges_file_path = os.path.join(edges_dir_path, f"{k}.csv")
                async with aiofiles.open(edges_file_path, "a") as csvf:
                    for edge in v:
                        await csvf.write(f"{edge[0]}\t{edge[1]}\n")

    async def _get_edges(
        self,
        local_authors: List[Dict[str, Any]],
        title: str,
        c_titles: List[str],
        domains: List[str],
    ):
        edges = defaultdict(list)
        paper_list = list(self.papers)
        paper_index = paper_list.index(title)
        authors = list(self.authors)
        domain = domains[0].split(".")[0]
        self.paper_domain[paper_index] = _DOMAINS.index(domain)
        for c_title in c_titles:
            citation_index = paper_list.index(c_title)
            self.paper_domain[citation_index] = _DOMAINS.index(domain)
            edges["paper__cites__paper"].append((paper_index, citation_index))
        for author in local_authors:
            halauthorid = author["halauthorid"]
            if halauthorid == "0":
                continue
            author_index = authors.index(halauthorid)
            local_institutions = author["affiliations"]
            for institution in local_institutions:
                edges["author__affiliated_with__institution"].append(
                    (author_index, self.institutions.index(institution))
                )
            edges["author__writes__paper"].append((author_index, paper_index))
        return edges

    async def _process(self, document: Dict[str, Any], path: str):
        # Get correct header
        halid = document["halid"]
        domains = document["domain"]
        header = self.headers[halid]

        # Process XML file with the header.
        async with aiofiles.open(path, "r") as xmlf:
            xml = await xmlf.read()

        root = helpers.str_to_xml(xml)

        # Get nodes
        title = header["title"]
        self._add_paper_node(title, halid)
        c_titles = self._get_citations(root)
        for c_title in c_titles:
            self._add_paper_node(c_title)
        local_authors = header["authors"]
        self._add_author_nodes(local_authors)
        self._add_institution_node(local_authors)

        # Get edges
        edges = await self._get_edges(
            local_authors=local_authors,
            title=title,
            c_titles=c_titles,
            domains=domains,
        )

        return edges

    def _add_paper_node(self, title: str, halid=""):
        if title not in self.papers:
            self.papers[title] = halid

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
            if halauthorid in self.authors and name not in self.authors[halauthorid]:
                self.authors[halauthorid].append(name)
            elif halauthorid not in self.authors:
                self.authors[halauthorid] = [name]

    async def build(self, output_dir_path: str):
        q = asyncio.Queue()
        await asyncio.gather(self._build(q), self._save_edges(q, output_dir_path))
        self.save_nodes(output_dir_path)
        edges_dir_path = os.path.join(output_dir_path, "edges")
        for file in os.listdir(edges_dir_path):
            file_path = os.path.join(edges_dir_path, file)
            helpers.gzip_compress(file_path)

    def save_nodes(self, output_dir_path: str):
        base_dir_path = helpers.check_dir(os.path.join(output_dir_path, "nodes"))
        # Save paper nodes
        papers_file_path = os.path.join(base_dir_path, "id_paper.csv.gz")
        with gzip.open(papers_file_path, "wt", encoding="utf-8") as csvf:
            for idx, (title, halid) in enumerate(self.papers.items()):
                csvf.write(f"{idx}\t{halid}\t{title}\n")
        # Save institutions nodes
        institutions_file_path = os.path.join(base_dir_path, "id_institution.csv.gz")
        with gzip.open(institutions_file_path, "wt", encoding="utf-8") as csvf:
            for idx, institution in enumerate(self.institutions):
                csvf.write(f"{idx}\t{institution}\n")
        # Save fields of study nodes
        labels_dir_path = helpers.check_dir(os.path.join(base_dir_path, "labels"))
        # Save label types
        domains_file_path = os.path.join(labels_dir_path, "domains.csv.gz")
        with gzip.open(domains_file_path, "wt", encoding="utf-8") as csvf:
            for idx, domain in enumerate(_DOMAINS):
                csvf.write(f"{idx}\t{domain}\n")
        # Save papers' labels
        author_domain_file_path = os.path.join(labels_dir_path, "paper__domain.csv.gz")
        with gzip.open(author_domain_file_path, "wt", encoding="utf-8") as csvf:
            for idx, (paper_idx, domain_idx) in enumerate(self.paper_domain.items()):
                csvf.write(f"{idx}\t{paper_idx}\t{domain_idx}\n")
        # Save author nodes
        authors_file_path = os.path.join(base_dir_path, "halid_authors.csv.gz")
        with gzip.open(authors_file_path, "wt", encoding="utf-8") as csvf:
            for idx, (halid, authornames) in enumerate(self.authors.items()):
                for authorname in authornames:
                    csvf.write(f"{idx}\t{halid}\t{authorname}\n")
