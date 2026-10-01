import logging
from pathlib import Path
import json
from typing import Any
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
import mlflow
import mlflow.transformers

logger = logging.getLogger(__name__)

CURATED_PARAMS = [
    "num_train_epochs",
    "learning_rate",
    "weight_decay",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "lr_scheduler_type",
    "warmup_ratio",
    "seed",
]


class CuratedMlflowCallback(TrainerCallback):
    """
    Custom MLflow logger to avoind the 500-character parameter truncation errors
    """

    def __init__(self, extra_params: dict | None = None) -> None:
        super().__init__()
        self.extra_params = extra_params or {}

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_world_process_zero:
            return

        args_dict = args.to_dict()
        params = {k: args_dict[k] for k in CURATED_PARAMS if k in args_dict}
        params.update(self.extra_params)

        # Truncate string representations safely
        safe_params = {k: str(v)[:500] for k, v in params.items()}
        mlflow.log_params(safe_params)

        # Dump full un-truncated config as artifact
        mlflow.log_dict(args_dict, "config/training_args.json")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if logs and state.is_world_process_zero:
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(metrics, step=state.global_step)


class ModelCardCallback(TrainerCallback):
    """
    Generate the model card and bundle with model artifacts at end of
    training to log the best checkpointed model to mlflow
    """

    def __init__(self, artifact_filename: str) -> None:
        super().__init__()
        self.artifact_filename = artifact_filename
        self.trainer: Trainer | None = None

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:

        if not state.is_world_process_zero:
            return

        trainer = self.trainer

        if trainer is None:
            logger.warning("Trainer reference was not set on callback")
            return

        meta_path = Path(args.output_dir) / self.artifact_filename
        readme_path = Path(args.output_dir) / "README.md"

        self.trainer.create_model_card(
            language="en",
            license="apache-2.0",
            tags=["text-generation", "lora", "azure-ml"],
            model_name="Qwen-Zookeeper-Producer",
            dataset="custom-jsonl",
        )

        metadata = self.create_final_metadata(args, state)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        mlflow.log_artifacts(args.output_dir, artifact_path="adapter_model")

        # mlflow.transformers.log_model(
        #     transformers_model={
        #         "model": trainer.model,
        #         "tokenizer": trainer.processing_class,
        #     },
        #     model_config=quant_config,
        #     artifact_path="best_model",
        #     artifacts=extra_artifacts,
        #     task="text-generation",
        # )

        logger.info(f"Updated best checkpoint metadata at: {meta_path}")

    def create_final_metadata(
        self,
        args: TrainingArguments,
        state: TrainerState,
    ) -> dict[str, int | str]:
        """
        Helper to create the custom model metadata artifact
        """
        return {
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
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_seq_length": getattr(args, "max_seq_length", None),
            },
        }
