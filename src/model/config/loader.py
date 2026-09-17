from pathlib import Path
import yaml
import logging

from model.config.model_config import ModelFitConfig

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> ModelFitConfig:
    """
    Loads and validates data prep config yaml
    """
    logger.info(f"Attempting to load config from {config_path}")
    path = Path(config_path)

    if not path.is_absolute():
        logger.info("Path is not absolute - creating absolute path to access")
        path = Path.cwd() / path

    if not path.exists():
        logger.error("Config file not found for data preparation config")
        raise FileNotFoundError("Config file not found for data preparation: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return ModelFitConfig.model_validate(data)
