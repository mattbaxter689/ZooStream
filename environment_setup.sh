#!/usr/bin/env bash

set -e

# 1. Load variables from .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# 2. Assign environment variables (or set default fallback values)
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-my-rg}"
WORKSPACE_NAME="${AZURE_WORKSPACE_NAME:-my-workspace}"

DATA_YAML_1="${DATA_YAML_1:-./infra/zoo_data.yaml}"
DATA_YAML_2="${DATA_YAML_2:-./infra/qwen_asset.yaml}"
ENV_PYTHON_SCRIPT="${ENV_PYTHON_SCRIPT:-./infra/create_environment.py}"

echo "=========================================="
echo "Target Workspace: ${WORKSPACE_NAME}"
echo "Resource Group:   ${RESOURCE_GROUP}"
echo "=========================================="

# 3. Create Data Asset 1
echo -e "\n[1/3] Registering Data Asset 1 from $DATA_YAML_1..."
az ml data create -f "$DATA_YAML_1" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE_NAME"

# 4. Create Data Asset 2
echo -e "\n[2/3] Registering Data Asset 2 from $DATA_YAML_2..."
az ml data create -f "$DATA_YAML_2" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE_NAME"

# 5. Create Environment via Python using uv
echo -e "\n[3/3] Creating Environment Asset via Python using uv..."
uv run "$ENV_PYTHON_SCRIPT"

echo -e "\n✓ All registrations completed successfully!"
