# halph/utils/data/link_prediction_metadata.py

import gc
import logging
import os
import os.path as osp
from typing import Any, Dict, Optional

import pandas as pd
from lxml import etree
from lxml.etree import ElementTree
from pandarallel import pandarallel

from halph.utils import helpers


class LinkPredictionMetadata:
    """Class.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------
    """

    df: pd.DataFrame

    def __init__(
        self,
        root_dir: str,
        json_dir: str,
        xml_dir: str,
    ):
        self.root_dir = helpers.check_dir(root_dir)
        self.raw_dir = helpers.check_dir(osp.join(root_dir, "raw"))
        helpers.check_dir(osp.join(self.raw_dir, "nodes"))
        helpers.check_dir(osp.join(self.raw_dir, "edges"))
        self.json_dir = json_dir
        self.xml_dir = xml_dir
        self.headers: Dict[str, Any] = {}
        os.makedirs("tmp", exist_ok=True)

    def __call__(self):
        pass

    @property
    def json_file_names(self):
        _json_file_names = os.listdir(self.json_dir)
        json_file_names = [
            json_file_name
            for json_file_name in _json_file_names
            if json_file_name.endswith(".json")
        ]
        return json_file_names

    @property
    def json_file_paths(self):
        json_file_paths = [
            osp.join(self.json_dir, json_file_name)
            for json_file_name in self.json_file_names
        ]
        return json_file_paths

    @property
    def xml_file_names(self):
        _xml_file_names = os.listdir(self.xml_dir)
        xml_file_names = [
            xml_file_name
            for xml_file_name in _xml_file_names
            if xml_file_name.endswith(".grobid.tei.xml")
        ]
        return xml_file_names

    def _get_citations(self, root: ElementTree):
        bibl_structs = root.xpath("//text/back/div/listBibl/biblStruct")
        c_titles = []
        c_years = []
        for bibl_struct in bibl_structs:
            c_title = self.get_citations_title(bibl_struct)
            c_year = self.get_citation_year(bibl_struct)
            if not c_title.split():
                continue
            c_titles.append(c_title)
            c_years.append(c_year)
        return c_titles, c_years

    def _compute_citations(self, df: pd.DataFrame):
        pandarallel.initialize(progress_bar=True)
        c_papers = pd.DataFrame()
        c_papers["cite"] = df["halid"].parallel_apply(self._worker)
        c_papers["halid"] = df["halid"]
        halid = c_papers["halid"]
        logging.info("Normalizing citations...")
        c_papers = pd.json_normalize(c_papers["cite"])
        logging.info("Flattening citations...")
        c_papers = c_papers.explode(["title", "year"])
        c_papers["c_halid"] = halid
        c_papers = c_papers.reset_index(drop=True)
        logging.info(c_papers)
        return c_papers

    def _worker(self, halid: str):
        xml_file_name = f"{halid}.grobid.tei.xml"
        if xml_file_name not in self.xml_file_names:
            return
        path = osp.join(self.xml_dir, xml_file_name)
        with open(path, "r") as xmlf:
            xml = xmlf.read()
        root = helpers.str_to_xml(xml)
        c_titles, c_years = self._get_citations(root)
        return {"title": c_titles, "year": c_years}

    def compute_edges(self, df: pd.DataFrame):
        logging.info("Computing edges...")
        path = osp.join(self.raw_dir, "nodes", "papers.csv.gz")
        papers = pd.read_csv(path, sep="\t", compression="gzip")
        path = osp.join(self.raw_dir, "nodes", "domains.csv.gz")
        domains = pd.read_csv(path, sep="\t", compression="gzip")
        path = osp.join(self.raw_dir, "nodes", "authors.csv.gz")
        authors = pd.read_csv(
            path, sep="\t", compression="gzip", dtype={"halauthorid": "object"}
        )
        path = osp.join(self.raw_dir, "nodes", "affiliations.csv.gz")
        affiliations = pd.read_csv(
            path, sep="\t", compression="gzip", dtype={"affiliations": "object"}
        )

        paper_domain = papers[["domain", "paper_idx"]]
        paper_domain["domain"] = paper_domain["domain"].apply(
            lambda x: x.strip("[]").replace("'", "").split(", ")
        )
        paper_domain = paper_domain.explode("domain")
        paper_domain["domain"] = paper_domain["domain"].apply(
            lambda x: x.split(".")[0] if x and isinstance(x, str) else ""
        )
        paper_domain = paper_domain[paper_domain.domain != ""]
        paper_domain = paper_domain.merge(domains, left_on="domain", right_on="domain")
        logging.info(paper_domain)
        paper_domain = paper_domain[["paper_idx", "domain_idx"]].drop_duplicates()
        path = osp.join(self.raw_dir, "edges", "paper__has_topic__domain.csv.gz")
        paper_domain.to_csv(path, compression="gzip", sep="\t", index=False)
        del paper_domain
        gc.collect()

        author_affiliation = df[df.halauthorid != "0"]
        author_affiliation = author_affiliation[["name", "halauthorid", "affiliations"]]
        author_affiliation = author_affiliation.explode("affiliations")
        author_affiliation = author_affiliation.merge(
            authors[["halauthorid", "author_idx"]],
            left_on="halauthorid",
            right_on="halauthorid",
        )
        author_affiliation = author_affiliation.merge(
            affiliations, left_on="affiliations", right_on="affiliations"
        )
        logging.info(author_affiliation)
        author_affiliation = author_affiliation[
            ["author_idx", "affiliation_idx"]
        ].drop_duplicates()
        path = osp.join(
            self.raw_dir, "edges", "author__affiliated_with__affiliation.csv.gz"
        )
        author_affiliation.to_csv(path, sep="\t", compression="gzip", index=False)
        del author_affiliation
        gc.collect()

        author_paper = df[["halid", "halauthorid"]]
        author_paper = author_paper[author_paper.halauthorid != "0"]
        logging.info(author_paper)
        author_paper = author_paper.merge(
            papers[["halid", "paper_idx"]], left_on="halid", right_on="halid"
        )
        author_paper = author_paper.merge(
            authors[["halauthorid", "author_idx"]],
            left_on="halauthorid",
            right_on="halauthorid",
        )
        logging.info(author_paper)
        author_paper = author_paper[["author_idx", "paper_idx"]].drop_duplicates()
        path = osp.join(self.raw_dir, "edges", "author__writes__paper.csv.gz")
        author_paper.to_csv(path, sep="\t", compression="gzip", index=False)
        del author_paper
        gc.collect()

        c_papers = self._compute_citations(papers)
        papers = (
            pd.concat([papers, c_papers], ignore_index=True)
            .drop_duplicates(subset=["title", "year"])
            .reset_index(drop=True)
        )
        papers["paper_idx"] = papers.index
        paper_paper = c_papers.merge(
            papers[["halid", "paper_idx"]],
            left_on="c_halid",
            right_on="halid",
            how="left",
        )
        to_ = c_papers.merge(
            papers[["title", "year", "paper_idx"]],
            left_on=["title", "year"],
            right_on=["title", "year"],
            how="left",
        )
        to_ = to_[["paper_idx"]].rename(columns={"paper_idx": "c_paper_idx"})
        paper_paper["c_paper_idx"] = to_
        paper_paper = paper_paper[["paper_idx", "c_paper_idx"]].dropna()
        paper_paper = paper_paper.astype({"paper_idx": int, "c_paper_idx": int})
        logging.info(paper_paper)
        path = osp.join(self.raw_dir, "edges", "paper_icites__paper.csv.gz")
        paper_paper.to_csv(path, sep="\t", compression="gzip", index=False)
        papers = papers.drop(["c_halid"], axis=1)
        path = osp.join(self.raw_dir, "nodes", "papers.csv.gz")
        papers.to_csv(path, sep="\t", compression="gzip", index=False)

    def compute_nodes(self, df: pd.DataFrame, lang: Optional[str] = None):
        logging.info("Computing nodes...")
        if lang is not None:
            logging.info(f"Computing nodes for {lang} language.")
            df.drop(df.loc[df["lang"] != lang].index, inplace=True)
            logging.info(df.info())
            logging.info(df.head())

        logging.info("Computing paper nodes...")
        path = osp.join(self.raw_dir, "nodes", "papers.csv.gz")
        df_ = df[["halid", "year", "title", "lang", "domain"]]
        df_ = df_.drop_duplicates(subset=["halid"]).reset_index(drop=True)
        df_["paper_idx"] = df_.index
        logging.info(df_.info())
        logging.info(df_.head())
        df_.to_csv(path, compression="gzip", index=False, sep="\t")

        logging.info("Computing author nodes...")
        path = osp.join(self.raw_dir, "nodes", "authors.csv.gz")
        df_ = df[df.halauthorid != "0"]
        df_ = (
            df_[["name", "halauthorid"]]
            .drop_duplicates(subset=["halauthorid"])
            .reset_index(drop=True)
        )
        df_["author_idx"] = df_.index
        logging.info(df_.info())
        logging.info(df_.head())
        df_.to_csv(path, compression="gzip", index=False, sep="\t")

        logging.info("Computing affiliation nodes...")
        path = osp.join(self.raw_dir, "nodes", "affiliations.csv.gz")
        df_ = df[["affiliations"]].explode("affiliations")
        df_ = df_.drop_duplicates().reset_index(drop=True)
        df_["affiliation_idx"] = df_.index
        logging.info(df_.info())
        logging.info(df_.head())
        df_.to_csv(path, compression="gzip", index=False, sep="\t")

        logging.info("Computing domain nodes...")
        path = osp.join(self.raw_dir, "nodes", "domains.csv.gz")
        df_ = df[["domain"]].explode("domain")
        df_["domain"] = df_["domain"].apply(
            lambda x: x.split(".")[0] if x and isinstance(x, str) else ""
        )
        df_ = df_.drop_duplicates().reset_index(drop=True)
        df_ = df_[df_.domain != ""]
        df_["domain_idx"] = df_.index
        logging.info(df_.info())
        logging.info(df_.head())
        df_.to_csv(path, compression="gzip", index=False, sep="\t")

    # @fasteners.interprocess_locked("tmp/nodes.lock")
    # def _node_worker(self, batch: Dict[str, List[str]]):
    #     for halid in tqdm(batch["halid"]):
    #         if halid not in self.headers:
    #             continue
    #         if f"{halid}.grobid.tei.xml" not in self.xml_file_names:
    #             continue

    #         header = self.headers[halid]
    #         title = header["title"]
    #         year = header["date"]

    #         # Process XML file with the header.
    #         path = osp.join(self.xml_dir, f"{halid}.grobid.tei.xml")
    #         with open(path, "r") as xmlf:
    #             xml = xmlf.read()
    #         root = helpers.str_to_xml(xml)

    #         # Save nodes
    #         path = osp.join(self.raw_dir, "nodes", "papers.csv")
    #         with open(path, "a", encoding="utf-8") as csvf:
    #             csvf.write(f"{halid}\t{year}\t{title.lower()}\n")
    #             c_titles, c_years = self._get_citations(root)
    #             halid_ = ""
    #             for c_title, c_year in zip(c_titles, c_years):
    #                 if not c_title.split():
    #                     continue
    #                 csvf.write(f"{halid_}\t{c_year}\t{c_title.lower()}\n")

    #         authors = header["authors"]
    #         author_path = osp.join(self.raw_dir, "nodes", "authors.csv")
    #         institution_path = osp.join(self.raw_dir, "nodes", "institutions.csv")
    #         f_author = open(author_path, "a", encoding="utf-8")
    #         f_institution = open(institution_path, "a", encoding="utf-8")

    #         for author in authors:
    #             halauthorid = author["halauthorid"]
    #             if halauthorid == "0":
    #                 continue
    #             f_author.write(f"{halauthorid}\t{author['name']}\n")
    #             for institution in author["affiliations"]:
    #                 f_institution.write(f"{institution}\n")

    #         f_author.close()
    #         f_institution.close()

    # def compute_nodes(self):
    #     self.headers = helpers.jsons_to_dict(self.json_file_paths, on="halid")
    #     processes = []
    #     batch_size = int(len(self.dataset) / self.num_proc) + 1
    #     logging.info("Initializing processes...")
    #     for process_nb in tqdm(range(self.num_proc)):
    #         begin_idx = process_nb * batch_size
    #         end_idx = begin_idx + batch_size
    #         batch = self.dataset[begin_idx:end_idx]
    #         p = multiprocessing.Process(target=self._node_worker, args=(batch,))
    #         p.start()
    #         processes.append(p)

    #     for p in processes:
    #         p.join()

    # def compute_edges(self):
    #     self.headers = helpers.jsons_to_dict(self.json_file_paths, on="halid")
    #     processes = []
    #     batch_size = int(len(self.dataset) / self.num_proc) + 1
    #     logging.info("Initializing processes...")
    #     for process_nb in tqdm(range(self.num_proc)):
    #         begin_idx = process_nb * batch_size
    #         end_idx = begin_idx + batch_size
    #         batch = self.dataset[begin_idx:end_idx]
    #         p = multiprocessing.Process(target=self._node_worker, args=(batch,))
    #         p.start()
    #         processes.append(p)

    #     for p in processes:
    #         p.join()

    # def deduplicate_nodes(self, compress: bool):
    #     path = osp.join(self.raw_dir, "nodes", "papers.csv")
    #     df = pd.read_csv(path, sep="\t", names=["halid", "year", "title"])
    #     df = df.groupby(df["title"], as_index=False).aggregate(
    #         {"halid": "max", "year": "max"}
    #     )
    #     df = df.dropna(subset=["title"], how="all").reset_index(drop=True)
    #     if compress:
    #         helpers.compress_csv(df, path)

    #     path = osp.join(self.raw_dir, "nodes", "authors.csv")
    #     df = pd.read_csv(path, sep="\t", names=["halauthorid", "year"])
    #     df = df.drop_duplicates().reset_index(drop=True)
    #     if compress:
    #         helpers.compress_csv(df, path)

    #     path = osp.join(self.raw_dir, "nodes", "institutions.csv")
    #     df = pd.read_csv(path, sep="\t", names=["institution"])
    #     df = df.drop_duplicates().reset_index(drop=True)
    #     if compress:
    #         helpers.compress_csv(df, path)

    @staticmethod
    def get_citations_title(bibl_struct: etree._Element):
        title_element = bibl_struct.find(".//title[@type='main']")
        title = title_element.text if title_element is not None else ""
        return title

    @staticmethod
    def get_citation_year(bibl_struct: etree._Element):
        date_element = bibl_struct.find(".//imprint/date[@type='published']")
        year = (
            date_element.attrib["when"].split("-")[0]
            if date_element is not None
            else "0"
        )
        return year

    # def __init__(self, dataset: DatasetDict, json_dir_path: str, xml_dir_path: str):
    #     super().__init__(
    #         template="link_prediction",
    #         json_dir_path=json_dir_path,
    #         xml_dir_path=xml_dir_path,
    #         dataset=dataset,
    #     )

    # def __call__(self, output_dir_path: str):
    #     loop = asyncio.get_event_loop()
    #     loop.run_until_complete(self.build(output_dir_path))

    # async def _build(self, q: asyncio.Queue):
    #     logging.info("Processing graph/text documents paired to headers...")
    #     for document in tqdm(self.dataset):
    #         halid = document["halid"]
    #         xml_file_path = os.path.join(self.xml_dir_path, f"{halid}.grobid.tei.xml")
    #         if xml_file_path not in self.xml_file_paths:
    #             continue
    #         if halid not in self.headers:
    #             continue
    #         edges = await self._process(document, xml_file_path)
    #         await q.put(edges)
    #     await q.put(None)

    # async def _process(self, document: Dict[str, Any], path: str):
    #     # Get correct header
    #     halid = document["halid"]
    #     domains = document["domain"]
    #     year = document["year"]
    #     header = self.headers[halid]

    #     # Process XML file with the header.
    #     async with aiofiles.open(path, "r") as xmlf:
    #         xml = await xmlf.read()

    #     root = helpers.str_to_xml(xml)

    #     # Get nodes
    #     title = header["title"]
    #     self._add_paper_node(title, halid, year)
    #     c_titles, c_years = self._get_citations(root)
    #     for c_title, c_year in zip(c_titles, c_years):
    #         self._add_paper_node(c_title, year=c_year)
    #     local_authors = header["authors"]
    #     self._add_author_nodes(local_authors)
    #     self._add_institution_node(local_authors)

    #     # Get edges
    #     edges = await self._get_edges(
    #         local_authors=local_authors,
    #         title=title,
    #         c_titles=c_titles,
    #         domains=domains,
    #     )

    #     return edges

    # async def _save_edges(self, q: asyncio.Queue, output_dir_path: str):
    #     edges_dir_path = helpers.check_dir(os.path.join(output_dir_path, "edges"))
    #     while True:
    #         edges = await q.get()
    #         if edges is None:
    #             break
    #         for k, v in edges.items():
    #             edges_file_path = os.path.join(edges_dir_path, f"{k}.csv")
    #             async with aiofiles.open(edges_file_path, "a") as csvf:
    #                 for edge in v:
    #                     await csvf.write(f"{edge[0]}\t{edge[1]}\n")

    # async def _get_edges(
    #     self,
    #     local_authors: List[Dict[str, Any]],
    #     title: str,
    #     c_titles: List[str],
    #     domains: List[str],
    # ):
    #     edges = defaultdict(list)
    #     paper_list = list(self.papers)
    #     paper_index = paper_list.index(title)
    #     authors = list(self.authors)
    #     try:
    #         domains_ = [self.domains.index(domain.split(".")[0]) for domain in domains]
    #         self.paper_domain[paper_index].extend(domains_)
    #     except ValueError:
    #         # No general domain provided for the paper
    #         domains_ = []
    #     for c_title in c_titles:
    #         citation_index = paper_list.index(c_title)
    #         self.paper_domain[citation_index].extend(domains_)
    #         edges["paper__cites__paper"].append((paper_index, citation_index))
    #     for author in local_authors:
    #         halauthorid = author["halauthorid"]
    #         if halauthorid == "0":
    #             continue
    #         author_index = authors.index(halauthorid)
    #         local_institutions = author["affiliations"]
    #         for institution in local_institutions:
    #             edges["author__affiliated_with__institution"].append(
    #                 (author_index, self.institutions.index(institution))
    #             )
    #         edges["author__writes__paper"].append((author_index, paper_index))
    #     return edges

    # def _add_paper_node(self, title: str, halid: str = "0", year: str = "0"):
    #     if title not in self.papers or (halid != "0" and title in self.papers[title]):
    #         self.papers[title] = (halid, year)

    # def _add_institution_node(self, authors: List[Dict[str, Any]]):
    #     for author in authors:
    #         institutions = author["affiliations"]
    #         for institution in institutions:
    #             if institution not in self.institutions:
    #                 self.institutions.append(institution)

    # def _add_author_nodes(self, authors: List[Dict[str, Any]]):
    #     for author in authors:
    #         halauthorid = author["halauthorid"]
    #         if halauthorid == "0":
    #             continue
    #         name = author["name"]
    #         if halauthorid in self.authors and name not in self.authors[halauthorid]:
    #             self.authors[halauthorid].append(name)
    #         elif halauthorid not in self.authors:
    #             self.authors[halauthorid] = [name]

    # async def build(self, output_dir_path: str):
    #     q = asyncio.Queue()
    #     await asyncio.gather(self._build(q), self._save_edges(q, output_dir_path))
    #     self.save_nodes(output_dir_path)
    #     edges_dir_path = os.path.join(output_dir_path, "edges")
    #     for file in os.listdir(edges_dir_path):
    #         file_path = os.path.join(edges_dir_path, file)
    #         helpers.gzip_compress(file_path)
    #         os.remove(file_path)

    # def save_nodes(self, output_dir_path: str):
    #     base_dir_path = helpers.check_dir(os.path.join(output_dir_path, "nodes"))
    #     # Save paper nodes
    #     papers_file_path = os.path.join(base_dir_path, "id_paper.csv.gz")
    #     with gzip.open(papers_file_path, "wt", encoding="utf-8") as csvf:
    #         for idx, (title, (halid, year)) in enumerate(self.papers.items()):
    #             csvf.write(f"{idx}\t{halid}\t{year}\t{title}\n")
    #     # Save institutions nodes
    #     institutions_file_path = os.path.join(base_dir_path, "id_institution.csv.gz")
    #     with gzip.open(institutions_file_path, "wt", encoding="utf-8") as csvf:
    #         for idx, institution in enumerate(self.institutions):
    #             csvf.write(f"{idx}\t{institution}\n")
    #     # Save author nodes
    #     authors_file_path = os.path.join(base_dir_path, "halauthorid_author.csv.gz")
    #     with gzip.open(authors_file_path, "wt", encoding="utf-8") as csvf:
    #         for idx, (halauthorid, authornames) in enumerate(self.authors.items()):
    #             for authorname in authornames:
    #                 csvf.write(f"{idx}\t{halauthorid}\t{authorname}\n")
    #     # Save domain nodes
    #     domains_file_path = os.path.join(base_dir_path, "domains.csv.gz")
    #     with gzip.open(domains_file_path, "wt", encoding="utf-8") as csvf:
    #         for idx, domain in enumerate(self.domains):
    #             csvf.write(f"{idx}\t{domain}\n")
    #     # Save domain edges
    #     author_domain_file_path = os.path.join(
    #         output_dir_path, "edges", "paper__has_topic__domain.csv"
    #     )
    #     with open(author_domain_file_path, "w", encoding="utf-8") as csvf:
    #         for idx, (paper_idx, domain_idxs) in enumerate(self.paper_domain.items()):
    #             for domain_idx in domain_idxs:
    #                 csvf.write(f"{paper_idx}\t{domain_idx}\n")
