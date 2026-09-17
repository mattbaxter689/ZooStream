from pydantic import BaseModel, ConfigDict, PositiveInt, Field


class PathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    output_dir: str
    artifact_name: str


class TrainConfig(BaseModel):
    max_seq_length: PositiveInt = 256
    epochs: PositiveInt = 10
    learning_rate: float = 2e-4
    batch_size: PositiveInt = 4
    gradient_accumulation_steps: PositiveInt = 4
    weight_decay: float = 0.01


class LoraConfigSchema(BaseModel):
    r: 16
    alpha: 32
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: list[str]


class ModelFitConfig(BaseModel):
    paths: PathsConfig
    train: TrainConfig
    lora: LoraConfigSchema
