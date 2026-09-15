from pydantic import BaseModel, ConfigDict, PositiveInt


class DataPrepPipelineConfig(BaseModel):
    """
    Central config for data preparation and processing
    """

    model_config = ConfigDict(frozen=True)
    output_name: str = "sft_transformed.jsonl"
    window_size: PositiveInt = 6
    stride: PositiveInt = 3
    max_samples: PositiveInt = 3000
    anomaly_keywords: tuple[str, ...] = (
        "ERROR",
        "WARN",
        "Exception",
        "ConnectionLoss",
        "SESSION EXPIRED",
        "FAILED",
    )
    log_regex: str = (
        r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+(\w+)\s+\[(.*)\]\s+-\s+(.*)$"
    )
