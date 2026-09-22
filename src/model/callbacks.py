import logging
import shutil
from pathlib import Path
import json
from typing import Any
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    Trainer,
)

logger = logging.getLogger(__name__)


class ArtifactSaveCallback(TrainerCallback):
    """
    Saves custom metadata to every checkpoint folder to ensure that the
    metadata I want is part of the associated data
    """

    def __init__(self, artifact_filename: str) -> None:
        super().__init__()
        self.artifact_filename = artifact_filename


class ModelCardCallback(TrainerCallback):
    """
    Generate the model card and bundle with model artifacts at end of
    training to log the best checkpointed model to mlflow
    """

    def __init__(self, artifact_filename: str) -> None:
        super().__init__()
        self.artifact_filename = artifact_filename
        self.loaded_artifacts: dict[str, Any] | None = None

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        metadata: dict[str, Any] = {
            "checkpoint_step": state.global_step,
            "best_metric_so_far": state.best_metric,
            "status": "Valid Checkpoint",
        }

        filepath = checkpoint_dir / self.artifact_filename
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Callback saved artifact to {filepath}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:

        best_checkpoint = state.best_model_checkpoint or args.output_dir
        trainer: Trainer | None = kwargs.get("trainer")

        if not best_checkpoint or not Path(best_checkpoint).exists():
            logger.warning(f"Best checkpoint not found: {best_checkpoint}")
            return

        logger.info(f"Best checkpoint verified: {best_checkpoint}")
        artifact_filepath = Path(best_checkpoint) / self.artifact_filename
        if artifact_filepath.exists():
            with open(artifact_filepath, "r") as f:
                self.loaded_artifacts = json.load(f)

        if trainer is None:
            return

        trainer.create_model_card(
            language="en",
            license="apache-2.0",
            tags=["text-generation", "lora", "azure-ml"],
            model_name="Qwen-Zookeeper-Producer",
            dataset="custom-jsonl",
        )
        source_readme = Path(args.output_dir) / "README.md"
        destination_readme = Path(best_checkpoint) / "README.md"
        if source_readme.exists() and source_readme != destination_readme:
            shutil.copy(source_readme, destination_readme)
            logger.info(f"Model card bundled to: {destination_readme}")
