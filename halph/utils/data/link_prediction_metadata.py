# halph/utils/data/link_prediction_metadata.py

import gc
import logging
import os
import os.path as osp
from typing import Any, Dict, List, Optional

import dask.dataframe as dd
import dask.dataframe.core as ddc
import pandas as pd
from dask.dataframe import from_pandas
from dask.diagnostics import ProgressBar
from lxml import etree
from lxml.etree import _ListErrorLog

from halph.utils import helpers


class LinkPredictionMetadata:

    def __init__(
        self,
        halids: List[str],
        root_dir: str,
        json_dir: str,
        xml_dir: str,
        raw_dir: str,
    ):
        self.halids = halids
        self.root_dir = helpers.check_dir(root_dir)
        self.raw_dir = helpers.check_dir(raw_dir)
        helpers.check_dir(osp.join(self.raw_dir, "nodes"))
        helpers.check_dir(osp.join(self.raw_dir, "edges"))
        self.json_dir = json_dir
        self.xml_dir = xml_dir

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
            and xml_file_name.split(".")[0]
            in self.halids  # Take only xml files with clean data
        ]
        return xml_file_names

    def _get_citations(self, root: etree._ElementTree):
        c_titles = []
        c_years = []
        bibl_structs = root.xpath("//text/back/div/listBibl/biblStruct")

        if isinstance(bibl_structs, _ListErrorLog):
            logging.error(f"Error with bibl struct.")
            return c_titles, c_years

        for bibl_struct in bibl_structs:
            try:
                c_title = self.get_citations_title(bibl_struct)
                c_year = self.get_citation_year(bibl_struct)
                if not c_title.split():
                    continue
                c_titles.append(str(c_title))
                c_years.append(str(c_year))
            except Exception as e:
                logging.error(e)
                continue
        return c_titles, c_years

    def _compute_citations(self, row):
        halid = row["halid"]
        c_title, c_year = self._worker(halid)
        return {"title": c_title, "year": c_year}

    def _worker(self, halid: str):
        xml_file_name = f"{halid}.grobid.tei.xml"
        c_titles = []
        c_years = []

        try:
            path = osp.join(self.xml_dir, xml_file_name)
            with open(path, "r") as xmlf:
                xml = xmlf.read()
        except FileNotFoundError:
            logging.error(f"File {xml_file_name} not found.")
            return c_titles, c_years

        try:
            root = helpers.str_to_xml(xml)
            c_titles, c_years = self._get_citations(root)
            return c_titles, c_years
        except Exception as e:
            logging.error(e)
            return c_titles, c_years

    def compute_edges(self, ddf: ddc.DataFrame, num_proc: int):
        logging.info("Computing edges.")

        # Loading raw nodes
        path = osp.join(self.raw_dir, "nodes", "papers.csv.gz")
        papers = pd.read_csv(path, sep="\t", compression="gzip", dtype={"halid": str})
        papers = papers.dropna(subset=["halid"])
        papers = from_pandas(papers, npartitions=num_proc)
        path = osp.join(self.raw_dir, "nodes", "domains.csv.gz")
        domains = pd.read_csv(path, sep="\t", compression="gzip")
        domains = from_pandas(domains, npartitions=num_proc)
        path = osp.join(self.raw_dir, "nodes", "authors.csv.gz")
        authors = pd.read_csv(path, sep="\t", compression="gzip")
        authors = from_pandas(authors, npartitions=num_proc)
        path = osp.join(self.raw_dir, "nodes", "affiliations.csv.gz")
        affiliations = pd.read_csv(path, sep="\t", compression="gzip")
        affiliations = from_pandas(affiliations, npartitions=num_proc)

        # Computing paper <-> domain raw edges
        logging.info("Computing paper <-> domain raw edges.")
        paper_domain = papers[["domain", "paper_idx"]]
        paper_domain = paper_domain.apply(self.str_to_list, axis=1, meta=paper_domain)
        paper_domain = paper_domain.explode("domain")
        paper_domain = paper_domain.apply(self.split_domain, axis=1, meta=paper_domain)
        paper_domain = paper_domain[paper_domain.domain != ""]
        paper_domain = paper_domain.merge(domains, left_on="domain", right_on="domain")
        paper_domain = paper_domain[["paper_idx", "domain_idx"]].drop_duplicates()
        logging.info(paper_domain)
        path = osp.join(self.raw_dir, "edges", "paper__has_topic__domain.csv.gz")
        paper_domain.to_csv(
            path, single_file=True, compression="gzip", sep="\t", index=False
        )
        del paper_domain
        gc.collect()

        # Computing author <-> affiliation raw edges
        logging.info("Computing author <-> affiliation raw edges.")
        author_affiliation = ddf[ddf.halauthorid != "0"]
        author_affiliation = author_affiliation[["name", "halauthorid", "affiliations"]]
        author_affiliation = author_affiliation.explode("affiliations").astype(
            {"halauthorid": float, "affiliations": float}
        )
        author_affiliation = author_affiliation.merge(
            authors[["halauthorid", "author_idx"]],
            left_on="halauthorid",
            right_on="halauthorid",
        )
        author_affiliation = author_affiliation.merge(
            affiliations, left_on="affiliations", right_on="affiliations"
        )
        author_affiliation = author_affiliation[
            ["author_idx", "affiliation_idx"]
        ].drop_duplicates()
        logging.info(author_affiliation)
        path = osp.join(
            self.raw_dir, "edges", "author__affiliated_with__affiliation.csv.gz"
        )
        author_affiliation.to_csv(
            path, single_file=True, sep="\t", compression="gzip", index=False
        )
        del author_affiliation
        gc.collect()

        # Computing author <-> paper raw edges
        logging.info("Computing author <-> paper raw edges.")
        author_paper = ddf[["halid", "halauthorid"]].astype(
            {"halauthorid": float, "halid": str}
        )
        author_paper = author_paper[author_paper.halauthorid != 0]
        author_paper = author_paper.merge(
            papers[["halid", "paper_idx"]],
            left_on="halid",
            right_on="halid",
        )
        author_paper = author_paper.merge(
            authors[["halauthorid", "author_idx"]],
            left_on="halauthorid",
            right_on="halauthorid",
        )
        author_paper = author_paper[["author_idx", "paper_idx"]].drop_duplicates()
        logging.info(author_paper)
        path = osp.join(self.raw_dir, "edges", "author__writes__paper.csv.gz")
        author_paper.to_csv(
            path, single_file=True, sep="\t", compression="gzip", index=False
        )
        del author_paper
        gc.collect()

        # Computing paper <-> paper raw edges
        c_papers = papers[["halid"]]
        c_papers_ = papers[["halid"]]
        with ProgressBar():
            c_papers_["cite"] = (
                c_papers_.apply(
                    self._compute_citations, axis=1, meta=("cite", "object")
                )
                .compute()
                .reset_index(drop=True)
            )
        with ProgressBar():
            c_papers_ = (
                c_papers_.map_partitions(
                    lambda x: pd.json_normalize(x["cite"].tolist()),
                    enforce_metadata=False,
                )
                .compute()
                .reset_index(drop=True)
            )
        c_papers = dd.concat([c_papers_, c_papers], axis=1, ignore_index=True)
        c_papers = c_papers.explode(["title", "year"]).reset_index(drop=True)

        papers = (
            dd.concat(
                [papers, c_papers[["title", "year"]]], sort=False, ignore_index=True
            )
            .drop_duplicates(subset=["title", "year"])
            .reset_index(drop=True)
        )
        papers["paper_idx"] = papers.index

        paper_paper = c_papers.merge(
            papers[["halid", "paper_idx"]],
            left_on="halid",
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
        paper_paper = dd.concat(
            [paper_paper, to_], axis=1, sort=False, ignore_index=True
        )
        paper_paper = paper_paper[["paper_idx", "c_paper_idx"]].dropna()
        paper_paper = paper_paper.astype({"paper_idx": int, "c_paper_idx": int})

        logging.info(paper_paper)
        path = osp.join(self.raw_dir, "edges", "paper__cites__paper.csv.gz")
        paper_paper.to_csv(
            path, single_file=True, sep="\t", compression="gzip", index=False
        )
        path = osp.join(self.raw_dir, "nodes", "papers.csv.gz")
        papers.to_csv(path, single_file=True, sep="\t", compression="gzip", index=False)

    def compute_nodes(
        self,
        ddf: ddc.DataFrame,
        langs: Optional[List[str]] = None,
        years: Optional[List[str]] = None,
    ):
        # Removing nodes from documents with no clean GROBID output
        ddf = ddf[ddf["halid"].isin(self.halids)].reset_index(drop=True)

        # Filtering based on list of languages
        if langs is not None:
            logging.info(f"Computing nodes for languages {' '.join(langs)}.")
            ddf = ddf[ddf["lang"].isin(langs)].reset_index(drop=True)
            logging.info(ddf)

        # Filtering based on list of years
        if years is not None:
            logging.info(f"Computing nodes for the years {' '.join(years)}.")
            ddf = ddf[ddf["year"].isin(years)].reset_index(drop=True)
            logging.info(ddf)

        # Computing paper nodes
        logging.info("Computing paper nodes")
        path = osp.join(self.raw_dir, "nodes", "papers.csv.gz")
        ddf_ = ddf[["halid", "year", "title", "lang", "domain"]]
        ddf_ = ddf_.drop_duplicates(subset=["halid"])
        ddf_ = ddf_[ddf_.title != ""].reset_index(drop=True)
        ddf_["paper_idx"] = ddf_.index
        logging.info(ddf_)
        ddf_.to_csv(path, single_file=True, compression="gzip", index=False, sep="\t")

        # Computing author nodes
        logging.info("Computing author nodes...")
        path = osp.join(self.raw_dir, "nodes", "authors.csv.gz")
        ddf_ = ddf[ddf.halauthorid != "0"]
        ddf_ = ddf_[["name", "halauthorid"]].drop_duplicates(subset=["halauthorid"])
        ddf_ = ddf_[ddf_.name != ""].reset_index(drop=True)
        ddf_["author_idx"] = ddf_.index
        logging.info(ddf_)
        ddf_.to_csv(path, single_file=True, compression="gzip", index=False, sep="\t")

        # Computing affiliation nodes
        logging.info("Computing affiliation nodes...")
        path = osp.join(self.raw_dir, "nodes", "affiliations.csv.gz")
        ddf_ = ddf[["affiliations"]].explode("affiliations")
        ddf_ = ddf_.drop_duplicates().reset_index(drop=True)
        ddf_["affiliation_idx"] = ddf_.index
        logging.info(ddf_)
        ddf_.to_csv(path, single_file=True, compression="gzip", index=False, sep="\t")

        # Computing domain nodes
        logging.info("Computing domain nodes...")
        path = osp.join(self.raw_dir, "nodes", "domains.csv.gz")
        ddf_ = (
            ddf[["domain"]]
            .explode("domain")
            .astype({"domain": str})
            .reset_index(drop=True)
        )
        ddf_ = ddf_.apply(self.split_domain, axis=1, meta=ddf_)
        ddf_ = ddf_.drop_duplicates().reset_index(drop=True)
        ddf_ = ddf_[ddf_.domain != ""]
        ddf_["domain_idx"] = ddf_.index
        logging.info(ddf_)
        ddf_.to_csv(path, single_file=True, compression="gzip", index=False, sep="\t")

    @staticmethod
    def split_domain(row):
        try:
            row["domain"] = row["domain"].split(".")[0]
        except:
            row["domain"] = "other"
        return row

    @staticmethod
    def str_to_list(row):
        try:
            row["domain"] = row["domain"].strip("[]").replace("'", "").split(", ")
        except:
            row["domain"] = []
        return row

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
