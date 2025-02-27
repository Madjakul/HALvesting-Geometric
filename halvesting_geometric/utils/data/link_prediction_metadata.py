# halvesting_geometric/utils/data/link_prediction_metadata.py

import gc
import logging
import os
import os.path as osp
from typing import List, Optional

import dask.dataframe as dd
import dask.dataframe.core as ddc
import pandas as pd
from dask.dataframe import from_pandas  # type: ignore
from dask.diagnostics import ProgressBar  # type: ignore
from lxml import etree

from halvesting_geometric.utils import helpers


class LinkPredictionMetadata:
    """Class to compute nodes and edges for link prediction tasks.

    Parameters
    ----------
    halids : List[str]
        List of HAL IDs to consider.
    root_dir : str
        Root directory where to store the raw data.
    json_dir : str
        Directory where to store the JSON files.
    xml_dir : str
        Directory where to store the XML files.
    raw_dir : str
        Directory where to store the raw data.

    Attributes
    ----------
    halids : List[str]
        List of HAL IDs to consider.
    root_dir : str
        Root directory where to store the raw data.
    json_dir : str
        Directory where to store the JSON files.
    xml_dir : str
        Directory where to store the XML files.
    raw_dir : str
        Directory where to store the raw data.
    json_file_names : List[str]
        List of JSON file names.
    json_file_paths : List[str]
        List of JSON file paths.
    xml_file_names : List[str]
        List of XML file names.

    Examples
    --------
    >>> from halvesting_geometric.utils.data import LinkPredictionMetadata
    >>> halids = ["00000001"]
    >>> root_dir = "data"
    >>> json_dir = "data/json"
    >>> xml_dir = "data/xml"
    >>> raw_dir = "data/raw"
    >>> metadata = LinkPredictionMetadata(halids, root_dir, json_dir, xml_dir, raw_dir)
    """

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
            if xml_file_name.endswith(".grobid.tei.xml") and xml_file_name.split(".")[0]
            # Take only xml files with clean data
            in self.halids
        ]
        return xml_file_names

    def _compute_citations(self, row: dd.Series):  # type: ignore
        """Compute citations for a given row.

        Parameters
        ----------
        row : dd.Series
            Row of the dataframe.

        Returns
        -------
        dict
            Dictionary containing the title and year of the citation.
        """
        halid = row["halid"]
        c_title, c_year = self._worker(halid)  # type: ignore
        return {"title": c_title, "year": c_year}

    def _get_citations(self, root: etree._ElementTree):
        """Get citations from a given XML file.

        Parameters
        ----------
        root : etree._ElementTree
            Root of the XML file.

        Returns
        -------
        c_titles : List[str]
            List of citation titles.
        """
        c_titles = []
        c_years = []
        bibl_structs = root.xpath("//text/back/div/listBibl/biblStruct")

        for bibl_struct in bibl_structs:
            try:
                c_title = self.get_citation_title(bibl_struct)
                c_year = self.get_citation_year(bibl_struct)
                if not c_title.split():
                    continue
                c_titles.append(str(c_title))
                c_years.append(str(c_year))
            except Exception as e:
                logging.error(e)
                continue
        return c_titles, c_years

    def _worker(self, halid: str):
        """Worker function to get citations from a given HAL ID.

        Parameters
        ----------
        halid : str
            HAL ID.

        Returns
        -------
        c_titles : List[str]
            List of citation titles.
        c_years : List[str]
            List of citation years.
        """
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
        """Compute edges for the link prediction task.

        Parameters
        ----------
        ddf : ddc.DataFrame
            Dask dataframe.
        num_proc : int
            Number of processes to use.
        """
        logging.info("Computing edges.")

        # Loading raw nodes
        path = osp.join(self.raw_dir, "nodes", "papers.csv")
        papers = pd.read_csv(path, sep="\t", dtype={"halid": str})
        papers = papers.dropna(subset=["halid"])
        papers = from_pandas(papers, npartitions=num_proc)
        path = osp.join(self.raw_dir, "nodes", "domains.csv")
        domains = pd.read_csv(path, sep="\t")
        domains = from_pandas(domains, npartitions=num_proc)
        path = osp.join(self.raw_dir, "nodes", "authors.csv")
        authors = pd.read_csv(path, sep="\t")
        authors = from_pandas(authors, npartitions=num_proc)
        path = osp.join(self.raw_dir, "nodes", "affiliations.csv")
        affiliations = pd.read_csv(path, sep="\t")
        affiliations = from_pandas(affiliations, npartitions=num_proc)

        # Computing paper <-> domain raw edges
        logging.info("Computing paper <-> domain raw edges.")
        paper_domain = papers[["domain", "paper_idx"]]
        paper_domain = paper_domain.apply(self.str_to_list, axis=1, meta=paper_domain)  # type: ignore
        paper_domain = paper_domain.explode("domain")
        paper_domain = paper_domain.apply(self.split_domain, axis=1, meta=paper_domain)
        paper_domain = paper_domain[paper_domain.domain != ""]
        paper_domain = paper_domain.merge(domains, left_on="domain", right_on="domain")
        paper_domain = paper_domain[["paper_idx", "domain_idx"]].drop_duplicates()
        logging.info(paper_domain)
        path = osp.join(self.raw_dir, "edges", "paper__has_topic__domain.csv")
        paper_domain.to_csv(path, single_file=True, sep="\t", index=False)
        del paper_domain
        gc.collect()

        # Computing author <-> affiliation raw edges
        logging.info("Computing author <-> affiliation raw edges.")
        author_affiliation = ddf[ddf.halauthorid != "0"]
        author_affiliation = author_affiliation[["name", "halauthorid", "affiliations"]]  # type: ignore
        author_affiliation = author_affiliation.explode("affiliations").astype(  # type: ignore
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
            self.raw_dir, "edges", "author__affiliated_with__affiliation.csv"
        )
        author_affiliation.to_csv(path, single_file=True, sep="\t", index=False)
        del author_affiliation
        gc.collect()

        # Computing author <-> paper raw edges
        logging.info("Computing author <-> paper raw edges.")
        author_paper = ddf[["halid", "halauthorid"]].astype(  # type: ignore
            {"halauthorid": float, "halid": str}
        )
        author_paper = author_paper[author_paper.halauthorid != 0]  # type: ignore
        author_paper = author_paper.merge(  # type: ignore
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
        path = osp.join(self.raw_dir, "edges", "author__writes__paper.csv")
        author_paper.to_csv(path, single_file=True, sep="\t", index=False)
        del author_paper
        gc.collect()

        # Computing paper <-> paper raw edges
        logging.info("Computing paper <-> paper raw edges.")
        c_papers = papers[["halid"]]
        c_papers_ = papers[["halid"]]
        # Get citations from grobid xml files
        with ProgressBar():
            c_papers_["cite"] = (  # type: ignore
                c_papers_.apply(  # type: ignore
                    self._compute_citations, axis=1, meta=("cite", "object")
                )
                .compute()
                .reset_index(drop=True)
            )
        # Normlaize citations
        with ProgressBar():
            c_papers_ = (
                c_papers_.map_partitions(  # type: ignore
                    lambda x: pd.json_normalize(x["cite"].tolist()),
                    enforce_metadata=False,
                )
                .compute()
                .reset_index(drop=True)
            )
        # Merge to get cited paper
        c_papers = dd.concat([c_papers_, c_papers], axis=1, ignore_index=True)  # type: ignore
        # Explode to get one row per citation
        c_papers = c_papers.explode(["title", "year"]).reset_index(drop=True)  # type: ignore
        # Add new papers to the papers dataframe
        papers = (
            dd.concat(  # type: ignore
                [papers, c_papers[["title", "year"]]], sort=False, ignore_index=True
            )
            .drop_duplicates(subset=["title", "year"])  # type: ignore
            .reset_index(drop=True)
        )
        papers["paper_idx"] = papers.index
        # Merge to get cited index
        paper_paper = c_papers.merge(
            papers[["title", "year", "paper_idx"]],
            left_on=["title", "year"],
            right_on=["title", "year"],
            how="left",
        ).reset_index(drop=True)
        paper_paper = paper_paper.rename(columns={"paper_idx": "c_paper_idx"})
        # Merge to get citing index
        paper_paper = paper_paper.merge(
            papers[["halid", "paper_idx"]], on="halid"
        ).reset_index(drop=True)
        paper_paper = paper_paper[["paper_idx", "c_paper_idx"]].astype(
            {"paper_idx": int, "c_paper_idx": int}
        )
        logging.info(paper_paper)
        path = osp.join(self.raw_dir, "edges", "paper__cites__paper.csv")
        paper_paper.to_csv(path, single_file=True, sep="\t", index=False)
        path = osp.join(self.raw_dir, "nodes", "papers.csv")
        papers.to_csv(path, single_file=True, sep="\t", index=False)

    def compute_nodes(
        self,
        ddf: ddc.DataFrame,
        langs: Optional[List[str]] = None,
        years: Optional[List[str]] = None,
    ):
        """Compute nodes for the link prediction task.

        Notes
        -----
        Getting the index from the domain nodes results in a wrong computation, yielding
        only 0 and 1. This occurs when the number of paper nodes is high. Make sure to
        check the domain nodes after the computation.

        Parameters
        ----------
        ddf : ddc.DataFrame
            Dask dataframe.
        langs : Optional[List[str]], optional
            Languages to consider, by default None.
        years : Optional[List[str]], optional
            Years to consider, by default None.
        """
        # Removing nodes from documents with no clean GROBID output
        ddf = ddf[ddf["halid"].isin(self.halids)].reset_index(drop=True)  # type: ignore

        # Filtering based on list of languages
        if langs is not None:
            logging.info(f"Computing nodes for languages {' '.join(langs)}.")
            ddf = ddf[ddf["lang"].isin(langs)].reset_index(drop=True)  # type: ignore
            logging.info(ddf)

        # Filtering based on list of years
        if years is not None:
            logging.info(f"Computing nodes for the years {' '.join(years)}.")
            ddf = ddf[ddf["year"].isin(years)].reset_index(drop=True)  # type: ignore
            logging.info(ddf)

        # Computing paper nodes
        logging.info("Computing paper nodes")
        path = osp.join(self.raw_dir, "nodes", "papers.csv")
        ddf_ = ddf[["halid", "year", "title", "lang", "domain"]]
        ddf_ = ddf_.drop_duplicates(subset=["halid"])  # type: ignore
        ddf_ = ddf_[ddf_.title != ""].reset_index(drop=True)
        ddf_["paper_idx"] = ddf_.index
        logging.info(ddf_)
        ddf_.to_csv(path, single_file=True, index=False, sep="\t")

        # Computing author nodes
        logging.info("Computing author nodes...")
        path = osp.join(self.raw_dir, "nodes", "authors.csv")
        ddf_ = ddf[ddf.halauthorid != "0"]
        ddf_ = ddf_[["name", "halauthorid"]].drop_duplicates(subset=["halauthorid"])  # type: ignore
        ddf_ = ddf_[ddf_.name != ""].reset_index(drop=True)
        ddf_["author_idx"] = ddf_.index
        logging.info(ddf_)
        ddf_.to_csv(path, single_file=True, index=False, sep="\t")

        # Computing affiliation nodes
        logging.info("Computing affiliation nodes...")
        path = osp.join(self.raw_dir, "nodes", "affiliations.csv")
        ddf_ = ddf[["affiliations"]].explode("affiliations")  # type: ignore
        ddf_ = ddf_.drop_duplicates().reset_index(drop=True)
        ddf_["affiliation_idx"] = ddf_.index
        logging.info(ddf_)
        ddf_.to_csv(path, single_file=True, index=False, sep="\t")

        # Computing domain nodes
        logging.info("Computing domain nodes...")
        path = osp.join(self.raw_dir, "nodes", "domains.csv")
        ddf_ = (
            ddf[["domain"]]
            .explode("domain")  # type: ignore
            .astype({"domain": str})
            .reset_index(drop=True)
        )
        ddf_ = ddf_.apply(self.split_domain, axis=1, meta=ddf_)
        ddf_ = ddf_.drop_duplicates().reset_index(drop=True)
        ddf_ = ddf_[ddf_.domain != ""].reset_index(drop=True)
        ddf_["domain_idx"] = ddf_.index
        logging.info(ddf_)
        ddf_.to_csv(path, single_file=True, index=False, sep="\t")

    @staticmethod
    def get_citation_title(bibl_struct: etree._Element):
        """Get the title of a citation.

        Parameters
        ----------
        bibl_struct : etree._Element
            Element of the XML file.

        Returns
        -------
        str
            Citation title.
        """
        title_element = bibl_struct.find(".//title[@type='main']")  # type: ignore
        title = title_element.text if title_element is not None else ""
        return title

    @staticmethod
    def get_citation_year(bibl_struct: etree._Element):
        date_element = bibl_struct.find(".//imprint/date[@type='published']")  # type: ignore
        year = (
            date_element.attrib["when"].split("-")[0]
            if date_element is not None
            else "0"
        )
        return year

    @staticmethod
    def split_domain(row: dd.Series):  # type: ignore
        """Apply function to split domain.

        Parameters
        ----------
        row : dd.Series
            Row of the dataframe.

        Returns
        -------
        dd.series
            Row of the dataframe.
        """
        try:
            row["domain"] = row["domain"].split(".")[0]  # type: ignore
        except:
            row["domain"] = "other"  # type: ignore
        return row

    @staticmethod
    def str_to_list(row: dd.Series):  # type: ignore
        """Apply function to convert string to list.

        Parameters
        ----------
        row : dd.series
            Row of the dataframe.

        Returns
        -------
        dd.series
            Row of the dataframe.
        """
        try:
            row["domain"] = row["domain"].strip("[]").replace("'", "").split(", ")  # type: ignore
        except:
            row["domain"] = []  # type: ignore
        return row
