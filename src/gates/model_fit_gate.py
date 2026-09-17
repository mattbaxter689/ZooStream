import argparse
from pathlib import Path

from utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model fit gate to train LLM model")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Location of the base Qwen model on Azure",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Location of transformed log data for model fit",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_fit_pipeline.yaml"),
        help="Path to yaml config for data prep pipeline",
    )

    return parser.parse_args()


def run(args: argparse.Namespace):

    setup_logging()


if __name__ == "__main__":
    args = parse_args()
    run(args)
