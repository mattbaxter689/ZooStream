set dotenv-load := true

# Submit DSL pipeline to Azure ML for training. For local submission only
run:
    uv run main.py

# Create the environment for Azure ML job runs
environment:
    uv run infra/create_environment.py
