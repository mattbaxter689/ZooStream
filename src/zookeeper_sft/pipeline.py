from collections.abc import Iterator
from pathlib import Path
import json
import logging

from zookeeper_sft.config.pipeline_config import DataPrepPipelineConfig
from zookeeper_sft.models import LogEntry, SFTRecord
from zookeeper_sft.parser import parse_line

logger = logging.getLogger(__name__)


def is_anomalous(window: list[LogEntry], keywords: tuple[str, ...]) -> bool:
    """
    Checks if a log entry in the window has the specified error level or
    keyword criteria
    """

    for entry in window:
        if entry.is_warning_or_error or any(kw in entry.raw for kw in keywords):
            return True
    return False


def get_sliding_windows(
    entries: list[LogEntry], size: int, stride: int
) -> Iterator[list[LogEntry]]:
    """
    Yields fixed length chunks using the configured window size and stride, defined via config
    """
    for i in range(0, len(entries) - size + 1, stride):
        yield entries[i : i + size]


def build_sft_dataset(
    config: DataPrepPipelineConfig, input_path: Path, output_path: Path
) -> int:
    if not input_path.exists():
        logger.error("Input file not found on Azure")
        raise FileNotFoundError("Input file from Azure does not exist")

    logger.info("Loading input zookeeper log data")
    with open(input_path, "r", encoding="utf-8") as f:

        logger.info("Parsing zookeeper logs to batches")
        entries = [
            parse_line(line.strip(), config.log_regex) for line in f if line.strip()
        ]

    output_path.mkdir(parents=True, exist_ok=True)
    records_written = 0

    with open(output_path / config.output_name, "w", encoding="utf-8") as f:

        logger.info("Parsing sliding window information")
        for window in get_sliding_windows(entries, config.window_size, config.stride):

            if records_written >= config.max_samples:
                break

            severity = (
                "ANOMALOUS_FAILURE"
                if is_anomalous(window, config.anomaly_keywords)
                else "NORMAL_OPERATIONAL"
            )
            logger.info(f"Constructing record {records_written + 1}")
            record = SFTRecord(
                instruction=(
                    f"Generate a sequence of {len(window)} ZooKeeper cluster log lines "
                    f"representing a {severity} state for component group '{window[0].primary_component}'."
                ),
                output="\n".join(e.raw for e in window),
            )

            f.write(json.dumps(record.to_dict()) + "\n")
            records_written += 1

        logger.info(f"Wrote {records_written} records")

    return records_written
