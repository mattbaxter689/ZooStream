import logging
from pathlib import Path
import sys

from azure.ai.ml import MLClient, dsl, command, Input, Output
from azure.ai.ml.entities import Command, PipelineJobSettings
from azure.identity import DefaultAzureCredential
from rich.console import Console
from rich.panel import Panel

from src.orchestrator.loader import load_orchestrator_config
from src.orchestrator.config import OrchestratorConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("orchestrator")
console = Console()

ROOT = Path(__file__).parent
CONFIG: OrchestratorConfig = load_orchestrator_config(
    "settings/orchestrator_config.yaml"
)


# ----- AML client and environment -------
def get_client() -> MLClient:
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=CONFIG.workspace.subscription_id,
        resource_group_name=CONFIG.workspace.resource_group,
        workspace_name=CONFIG.workspace.workspace_name,
    )


# ------ Base Job Config ----------
def _base_job_kwargs(name: str, description: str) -> dict:

    return {
        "display_name": name,
        "description": description,
        "environment": CONFIG.environment.full_name,
        "compute": CONFIG.compute.compute_cluster,
        "experiment_name": CONFIG.pipeline.experiment_name,
        "code": "./src",
        "environment_variables": {"PYTHONPATH": "./"},
    }


# ------ Job Builders ---------
# This is scaffolding from a previous project that is being resued
def data_prep_component() -> Command:
    return command(
        **_base_job_kwargs(
            "data-prep-gate", "Fetch, clean, and prepare data for training"
        ),
        command=(
            "python -m gates.data_prep_gate "
            "--raw-data ${{inputs.raw_data}} "
            "--output-path ${{outputs.processed_data}}"
        ),
        inputs={
            "raw_data": Input(type="uri_file", path=CONFIG.data.name),
        },
        outputs={
            "processed_data": Output(
                type="uri_folder",
                mode="rw_mount",
                name=CONFIG.data.output_asset_name,
            )
        },
    )


@dsl.pipeline(
    description="Gated classification pipeline",
    experiment_name=CONFIG.pipeline.experiment_name,
    default_compute=CONFIG.compute.compute_cluster,
)
def build_pipeline(raw_data: Input):
    prep_step = data_prep_component()(raw_data=raw_data)

    return prep_step.outputs


def main() -> None:
    ml_client = get_client()

    pipeline_job = build_pipeline(
        raw_data=Input(type="uri_file", path=CONFIG.data.name),
    )
    pipeline_job.settings = PipelineJobSettings(
        force_rerun=True, default_compute=CONFIG.compute.compute_cluster
    )

    try:
        returned_job = ml_client.jobs.create_or_update(pipeline_job)
        job_url = returned_job.studio_url
        console.print("\n[bold cyan]▶ Pipeline submitted[/bold cyan]")
        console.print(f"  Job ID : {returned_job.name}")
        console.print(f"  Studio : [link={job_url}]{job_url}[/link]")

        # Stream logs — blocks until pipeline completes
        ml_client.jobs.stream(returned_job.name)

        console.print(Panel("[bold green]Pipeline complete[/bold green]"))

    except RuntimeError as exc:
        console.print(
            Panel(f"[bold red]Pipeline halted:\n{exc}[/bold red]", title="FAILED")
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
