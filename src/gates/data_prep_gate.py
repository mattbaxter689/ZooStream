import logging
import argparse


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data prep gate to convert raw logs to useable output"
    )
    parser.add_argument(
        "--raw-data", type=str, required=True, help="Mounted input path to raw data"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="The output path to write the transformed data",
    )

    return parser.parse_args()
