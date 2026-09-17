from datasets import load_dataset
from pathlib import Path


def load_and_prepare_dataset(
    data_path: str | Path, test_size: float = 0.2, seed: int = 42
):
    data = load_dataset("json", data_files={"train": data_path})

    return data["train"].train_test_split(test_size=test_size, seed=seed)
