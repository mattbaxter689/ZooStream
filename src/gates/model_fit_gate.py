import argparse


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

    return parser.parse_args()


def run(args: argparse.Namespace):
    pass


if __name__ == "__main__":
    args = parse_args()
    run(args)
