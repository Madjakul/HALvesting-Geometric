# halph/utils/argparsers/build_matadata_argparse.py

import argparse

from halph.utils.helpers import boolean


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
            help="Name of the HuggingFace dataset containing the filtered documents.",
        )
        parser.add_argument(
            "--json_dir",
            type=str,
            default="./data/responses",
            help="Path to the HuggingFace cache directory.",
        )
        parser.add_argument(
            "--xml_dir",
            type=str,
            default="./data/output_tei_xml",
            help="Path to the txt file containing the dataset configs to process.",
        )
        parser.add_argument(
            "--compute_nodes",
            type=boolean,
            required=True,
            help="Path to the directory where the processed dataset will be saved.",
        )
        parser.add_argument(
            "--lang",
            type=str,
            default=None,
            help="Path to the directory where the processed dataset will be saved.",
        )
        parser.add_argument(
            "--compute_edges",
            type=boolean,
            required=True,
            help="Path to the directory where the processed dataset will be saved.",
        )
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Cache directory used to store downloaded HuggingFace datasets.",
        )
        args, _ = parser.parse_known_args()
        return args
