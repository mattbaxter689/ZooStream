import logging
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


class ModelCardCallback(TrainerCallback):
    """
    Generate the model card and bundle with model artifacts at end of
    training to log the best checkpointed model to mlflow
    """

    def __init__(self, artifact_filename: str) -> None:
        super().__init__()
        self.artifact_filename = artifact_filename
        self.loaded_artifacts: dict[str, Any] | None = None

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:

        if not args.should_save:
            return

        trainer: Trainer | None = kwargs.get("trainer")
        if trainer is None:
            logger.warning("Trainer instance not passed to callback kwargs")
            return

        trainer.create_model_card(
            language="en",
            license="apache-2.0",
            tags=["text-generation", "lora", "azure-ml"],
            model_name="Qwen-Zookeeper-Producer",
            dataset="custom-jsonl",
        )

        metadata = {
            "model_name": "Qwen-Zookeeper-Producer",
            "best_checkpoint_step": state.best_model_checkpoint,
            "best_metric": {
                "metric_name": args.metric_for_best_model,
                "value": state.best_metric,
            },
            "training_stats": {
                "epoch": state.epoch,
                "global_step": state.global_step,
                "total_flos": state.total_flos,
                "log_history": state.log_history[-5:],  # Last few logs
            },
            "hyperparameters": {
                "learning_rate": args.learning_rate,
                "batch_size": args.per_device_train_batch_size
                * args.gradient_accumulation_steps,
                "max_seq_length": getattr(args, "max_seq_length", None),
            },
        }
        meta_path = Path(args.output_dir) / self.artifact_filename
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Updated best checkpoint metadata at: {meta_path}")
