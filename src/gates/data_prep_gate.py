import logging
import argparse
from pathlib import Path
import sys

from utils.logging_utils import setup_logging
from zookeeper_sft.config.loader import load_config
from zookeeper_sft.pipeline import build_sft_dataset


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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_prep_pipeline.yaml"),
        help="Path to yaml config for data prep pipeline",
    )

    return parser.parse_args()


def run(args: argparse.Namespace):

    setup_logging()

    logger.info("Beginning data preparation gate")

    try:
        logger.info("Creating input paths for raw data and transformed")
        raw_path = Path(args.raw_data)
        output_path = Path(args.output_path)

        config = load_config(args.config)
        record_count = build_sft_dataset(config, raw_path, output_path)

        logger.info(f"Pipeline complete. Successfully exported {record_count} ")

    except Exception as e:
        logger.error(f"Error executing pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    run(args)
