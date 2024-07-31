# halvesting_geometric/utils/argparsers/build_matadata_argparse.py

import argparse

from halvesting_geometric.utils.helpers import boolean


class BuildMetadataArgparse:
    """Argument parser used to build the graphs' metadata."""

    @classmethod
    def parse_known_args(cls):
        """Parses arguments.

        Returns
        -------
        args: Any
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(
            description="Argument parser used to build the graphs' metadata."
        )
        parser.add_argument(
            "--root_dir",
            type=str,
            default="./data",
            help="Path to the directory where the processed dataset will be saved.",
        )
        parser.add_argument(
            "--json_dir",
            type=str,
            default="./data/responses",
            help="Path to the directyory where the JSON responses are stored.",
        )
        parser.add_argument(
            "--xml_dir",
            type=str,
            default="./data/output_tei_xml",
            help="Path to the directory where the TEI XML files are stored.",
        )
        parser.add_argument(
            "--raw_dir",
            type=str,
            required=True,
            help="Path to the raw computed graph.",
        )
        parser.add_argument(
            "--compute_nodes",
            type=boolean,
            required=True,
            help="Weather to compute the nodes or not.",
        )
        parser.add_argument(
            "--lang",
            type=str,
            default=None,
            help="Language of the documents to consider.",
        )
        parser.add_argument(
            "--compute_edges",
            type=boolean,
            required=True,
            help="Weather to compute the edges or not.",
        )
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Path to the directory where the dataset will be cached.",
        )
        parser.add_argument(
            "--dataset_checkpoint",
            type=str,
            default="Madjakul/HALvest-Geometric",
            help="Name of the HuggingFace dataset containing the filtered documents.",
        )
        parser.add_argument(
            "--dataset_config_path",
            type=str,
            default="./configs/dataset_config.txt",
            help="Path to the file containing the dataset configurations to download.",
        )
        parser.add_argument(
            "--zip_compress",
            type=boolean,
            default=False,
            help="Weather to compress the metadata or not.",
        )
        args, _ = parser.parse_known_args()
        return args
