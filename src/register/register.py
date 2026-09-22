import logging
import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.contants import AssetTypes
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)


def register_model_asset(checkpoint_path: str, model_name: str) -> None:
    """
    Registers trained model checkpoint from model fit to Azure ML workspace
    """

    logger.info(f"Registering model asset from path: {checkpoint_path}")
    try:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=os.environ.get("AZUREML_ARM_SUBSCRIPTION"),
            resource_group_name=os.environ.get("AZUREML_ARM_RESOURCEGROUP"),
            workspace_name=os.environ.get("AZUREML_ARM_WORKSPACE_NAME"),
        )

        model_asset = Model(
            path=checkpoint_path,
            type=AssetTypes.CUSTOM_MODEL,
            name=model_name,
            description="Qwen standard LoRA fine-tuned model asset with model card and custom metadata.",
            tags={
                "framework": "transformers",
                "method": "lora",
                "task": "text-generation",
            },
        )

        registered_model = ml_client.models.create_or_update(model_asset)
        logger.info(
            f"Successfully registered model {registered_model.name} version {registered_model.version}"
        )

    except Exception as e:
        logger.error(f"Error registering model asset: {e}")
        raise
