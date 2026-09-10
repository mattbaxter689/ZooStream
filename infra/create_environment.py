import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ.get("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.environ.get("AZURE_RESOURCE_GROUP"),
    workspace_name=os.environ.get("AZURE_WORKSPACE_NAME"),
)

env_docker_context = Environment(
    build=BuildContext(path=".", dockerfile_path="infra/Dockerfile"),
    name="log_generate",
    description="Environment created from Docker context for log generator",
)

ml_client.environments.create_or_update(env_docker_context)
