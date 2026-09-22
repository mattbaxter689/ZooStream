from trl import SFTTrainer, SFTConfig
import os
from pathlib import Path
import logging
from transformers import EarlyStoppingCallback

from model.config.loader import load_config
from model.data import load_and_prepare_dataset
from model.zoo_model import setup_model_and_tokenizer
from model.callbacks import ModelCardCallback

logger = logging.getLogger(__name__)


def train(
    config_path: str | Path,
    dataset_path: str | Path,
    model_path: str,
) -> str | Path:

    # load training configs
    cfg = load_config(config_path)

    step_run_id = os.environ.get("AZUREML_RUN_ID")
    if step_run_id:
        os.environ["MLFLOW_RUN_ID"] = step_run_id

    # load dataset
    split_dataset = load_and_prepare_dataset(dataset_path)

    model, tokenizer = setup_model_and_tokenizer(
        model_path=model_path,
        lora_r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
    )

    training_args = SFTConfig(
        output_dir=cfg.paths.output_dir,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        num_train_epochs=cfg.training.epochs,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        weight_decay=cfg.training.weight_decay,
        logging_steps=5,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="mlflow",
        dataset_text_field="text",
        max_seq_length=cfg.training.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        callbacks=[
            ModelCardCallback(cfg.paths.artifact_name),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    logger.info("Starting training from YAML config")
    trainer.train()

    # Save best model weights, config, and tokenizer to output output_dir
    trainer.save_model(cfg.paths.output_dir)
    tokenizer.save_pretrained(cfg.paths.output_dir)

    logger.info(f"Exported best model asset to {cfg.paths.output_dir}")

    return Path(cfg.paths.output_dir)
