# halvesting_geometric/utils/argparsers/link_prediction_argparse.py

import argparse


class LinkPredictionArgparse:
    """Argument parser used to perfomr link prediction."""

    @classmethod
    def parse_known_args(cls):
        """Parses arguments.

        Returns
        -------
        args: Any
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(
            description="""Aggregate the number of documents and tokens per language \
                and per domain."""
        )
        parser.add_argument(
            "--gnn",
            type=str,
            required=True,
            help="Name of message passing model used {'sage', 'gat', 'rggc'}.",
        )
        parser.add_argument(
            "--num_proc",
            type=int,
            default=4,
            help="Number of processes to use for processing the dataset.",
        )
        parser.add_argument(
            "--run",
            type=int,
            required=True,
            help="Number of processes to use for processing the dataset.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=4,
            help="Number of documents loaded per proc.",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            required=True,
            help="Set to `true` if the dataset has a `token_count` attribute.",
        )
        args, _ = parser.parse_known_args()
        return args
