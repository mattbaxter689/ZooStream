from trl import SFTTrainer, SFTConfig
import os
from pathlib import Path
import logging
from transformers import EarlyStoppingCallback

from model.config.loader import load_config
from model.data import load_and_prepare_dataset
from model.zoo_model import setup_model_and_tokenizer
from model.callbacks import ModelCardCallback, CuratedMlflowCallback

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
    dataset = f"{dataset_path.rstrip('/')}/sft_transformed.jsonl"
    split_dataset = load_and_prepare_dataset(dataset)

    model, tokenizer = setup_model_and_tokenizer(
        model_path=model_path,
        lora_r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
    )

    def to_prompt_completion(example):
        return {
            "prompt": [{"role": "user", "content": example["instruction"]}],
            "completion": [{"role": "assistant", "content": example["output"]}],
        }

    split_dataset = split_dataset.map(
        to_prompt_completion,
        remove_columns=["instruction", "output"],
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
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        max_length=cfg.training.max_seq_length,
        completion_only_loss=True,
        eos_token="<|im_end|>",
        disable_tqdm=True,
    )

    modelcard_callback = ModelCardCallback(cfg.paths.artifact_name)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        callbacks=[
            modelcard_callback,
            CuratedMlflowCallback(),
            EarlyStoppingCallback(early_stopping_patience=1),
        ],
    )
    modelcard_callback.trainer = trainer

    logger.info("Starting training from YAML config")
    trainer.train()

    trainer.save_model(cfg.paths.output_dir)
    tokenizer.save_pretrained(cfg.paths.output_dir)
    logger.info(f"Exported best model asset to {cfg.paths.output_dir}")

    return Path(cfg.paths.output_dir)
