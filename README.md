# ZooStream
 
Fine-tuning a small LLM (Qwen 2.5 Coder 1.5B Instruct) on real Apache Zookeeper log data, orchestrated as an Azure ML pipeline, so it can generate/stream realistic example error logs on demand via an API.
 
> **Status:** early scaffolding / work in progress. Config, infra, and data-asset registration are in place; the actual pipeline steps (data versioning, training, serving) are stubbed out and not yet implemented. See [Project Status](#project-status) below.
 
## Overview
 
The goal of this project is to take the [Zookeeper log dataset](dataset/Zookeeper_2k.log) (2k lines of real Zookeeper server logs), use it to fine-tune Qwen 2.5 Coder 1.5B Instruct, and expose the tuned model behind an API that can stream synthetic-but-realistic Zookeeper error logs — useful for testing log pipelines, alerting rules, or observability tooling without needing a live Zookeeper cluster misbehaving on demand.
 
Training and pipeline orchestration run on **Azure Machine Learning**, with hyperparameter search handled by **Optuna**.
 
## Project structure
 
```
.
├── main.py                       # Pipeline entrypoint (AML client + DSL pipeline) — WIP
├── dataset/
│   └── Zookeeper_2k.log          # Raw Zookeeper log sample used for tuning
├── infra/
│   ├── Dockerfile                # Container image for the AML training environment
│   ├── create_environment.py     # Registers the Docker-based AML Environment asset
│   ├── qwen_asset.yaml           # AML data asset definition for the base Qwen model weights
│   └── zoo_data.yaml             # AML data asset definition for the Zookeeper log dataset
├── settings/
│   └── orchestrator_config.yaml  # Workspace / compute / pipeline settings (env-var driven)
├── src/config/
│   ├── loader.py                 # YAML loaders with ${ENV_VAR} substitution
│   ├── orchestrator_config.py    # Pydantic models for orchestrator_config.yaml
│   ├── training_config.py        # Pydantic models for Optuna hyperparameter search space
│   └── training_config.yaml      # Optuna trial / hyperparameter definitions
├── environment_setup.sh          # One-shot script to register AML data assets + environment
├── justfile                      # `just run` — submit the pipeline locally
├── pyproject.toml                # Project metadata & dependencies (uv-managed)
└── uv.lock
```
 
## Prerequisites
 
- Python 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- [just](https://github.com/casey/just) (optional, for the `justfile` shortcuts)
- An Azure subscription with an [Azure Machine Learning workspace](https://learn.microsoft.com/azure/machine-learning/) and a GPU compute cluster. This project utilizes a previous created instance. The proper terraform files for setup an IAM orchestration, as well as compute cluster setup can be found [in this repository](https://github.com/mattbaxter689/AzureGates)
- Azure CLI (`az`), logged in and with the `ml` extension installed, if you plan to run `environment_setup.sh`
- Local weights for `qwen-2.5-coder-1.5b-instruct` sitting two directories above the repo root (see `infra/qwen_asset.yaml`), or an adjusted path pointing at wherever you've downloaded them
## Setup
 
1. **Clone and install dependencies**
```bash
   git clone https://github.com/mattbaxter689/ZooStream.git
   cd ZooStream
   uv sync
```
 
2. **Configure environment variables**
   Create a `.env` file in the repo root with your Azure ML workspace details:
```bash
   AZURE_SUBSCRIPTION_ID=<your-subscription-id>
   AZURE_RESOURCE_GROUP=<your-resource-group>
   AZURE_WORKSPACE_NAME=<your-workspace-name>
```
 
   These are consumed by `settings/orchestrator_config.yaml` via `src/config/loader.py`'s `${VAR}` substitution.
 
3. **Register data assets and the training environment on Azure ML**
```bash
   ./environment_setup.sh
```
**BE AWARE** Trying to load a HuggingFace model downloaded locally will be uploading multiple *gigabytes* at a time and can throttle your internet. To get around this, you can access Azure Cloud Shell. It has the HuggingFace CLI already configured, you just need to download the model and register the asset to your workspace in Azure ML.
 
   This registers two AML data assets (the Zookeeper log dataset and the Qwen model weights) from `infra/zoo_data.yaml` and `infra/qwen_asset.yaml`, then builds and registers the training `Environment` from `infra/Dockerfile` via `infra/create_environment.py`.
 
4. **Run the pipeline**
```bash
   just run
   # equivalent to: uv run main.py
```
 
## Configuration
 
- **`settings/orchestrator_config.yaml`** — workspace, compute cluster, experiment name, and the registered data/model asset names used by the AML pipeline. Validated against `src/config/orchestrator_config.py`.
- **`src/config/training_config.yaml`** — Optuna search space (trial count, epochs, hyperparameter ranges) for the fine-tuning run. Validated against `src/config/training_config.py`.
## Project Status
 
This repo is under active early development. Currently in place:
 
- ✅ Config loading/validation (Pydantic models + env-var substitution)
- ✅ Azure ML data asset and environment definitions
- ✅ Docker training environment
- ✅ Optuna hyperparameter search space definition
Not yet implemented (stubbed in `main.py`):
 
- ⬜ Data versioning/preprocessing pipeline step
- ⬜ Fine-tuning/training pipeline step
- ⬜ Full DSL pipeline wiring and submission
- ⬜ The API for streaming generated example logs
## License
 
No license has been specified yet for this repository.
