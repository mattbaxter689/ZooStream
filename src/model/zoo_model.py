import logging
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model

logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(
    model_path: Path | str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
) -> tuple[PeftModel, AutoTokenizer]:
    """
    Loads model in FP16 and wraps with the proper PEFT/LoRA adapters for LoRA
    training. QLoRA could also be used as well
    """
    logger.info(f"Loading tokenizer from path {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Loading FP16 base model from path {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    logger.info("Creating LoRA model config")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer
