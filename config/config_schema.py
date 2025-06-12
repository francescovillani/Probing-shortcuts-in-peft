from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import yaml


class PEFTConfig(BaseModel):
    """PEFT configuration settings"""
    peft_type: str = Field("none", description="Type of PEFT to use")
    peft_args: Dict[str, Any] = Field(default_factory=dict, description="PEFT-specific arguments")

    @field_validator("peft_type")
    @classmethod
    def validate_peft_type(cls, v):
        allowed_types = ["none", "lora", "qlora", "ia3", "prompt_tuning", 
                        "prefix_tuning", "p_tuning", "bitfit"]
        if v not in allowed_types:
            raise ValueError(f"peft_type must be one of {allowed_types}")
        return v


class DatasetConfig(BaseModel):
    """Dataset configuration settings"""
    name: str = Field(..., description="Dataset name or path")
    config: Optional[str] = Field(None, description="Dataset configuration name")
    batch_size: int = Field(..., description="Batch size")
    is_local: bool = Field(False, description="Whether dataset is local")
    split: Optional[str] = Field(None, description="Dataset split to use")

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError("Batch size must be greater than 0")
        return v


class WandBConfig(BaseModel):
    """Weights & Biases configuration"""
    project: str = Field("Naturally Occurring Shortcuts", description="WandB project name")
    enabled: bool = Field(True, description="Whether to use WandB logging")


class TrainingConfig(BaseModel):
    """Main training configuration"""
    # Model and Training
    modelname: str = Field(..., description="HuggingFace model name or path")
    num_labels: int = Field(..., description="Number of output labels")
    epochs: int = Field(..., description="Number of training epochs")
    lr: float = Field(..., description="Learning rate")
    seed: int = Field(42, description="Random seed")
    outputdir: str = Field("outputs", description="Output directory")

    # PEFT Configuration
    peft: PEFTConfig = Field(default_factory=PEFTConfig)

    # Dataset Configuration
    train_dataset: DatasetConfig
    validation_datasets: Dict[str, DatasetConfig]
    max_train_size: Optional[int] = None

    # Advanced Training Options
    tokenizer_max_length: int = Field(512, description="Maximum sequence length")
    gradient_accumulation_steps: int = Field(1, description="Gradient accumulation steps")
    warmup_ratio: float = Field(0.06, description="Warmup ratio for scheduler")

    # Checkpointing and Evaluation
    save_strategy: Literal["epoch", "steps"] = Field("epoch", description="When to save checkpoints")
    evaluation_strategy: Literal["epoch", "steps"] = Field("epoch", description="When to run evaluation")
    save_total_limit: Optional[int] = Field(None, description="Maximum number of checkpoints to keep")
    metric_for_best_model: str = Field("accuracy", description="Metric to use for best model selection")

    # Logging Configuration
    wandb: WandBConfig = Field(default_factory=WandBConfig)

    @field_validator("outputdir")
    @classmethod
    def create_output_dir(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @field_validator("lr")
    @classmethod
    def validate_learning_rate(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("Learning rate must be between 0 and 1")
        return v

    @field_validator("epochs")
    @classmethod
    def validate_epochs(cls, v):
        if v <= 0:
            raise ValueError("Epochs must be greater than 0")
        return v

    @field_validator("num_labels")
    @classmethod
    def validate_num_labels(cls, v):
        if v <= 0:
            raise ValueError("Number of labels must be greater than 0")
        return v

    @field_validator("warmup_ratio")
    @classmethod
    def validate_warmup_ratio(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Warmup ratio must be between 0 and 1")
        return v

    @field_validator("gradient_accumulation_steps")
    @classmethod
    def validate_gradient_accumulation_steps(cls, v):
        if v <= 0:
            raise ValueError("Gradient accumulation steps must be greater than 0")
        return v

    class Config:
        extra = "forbid"  # Prevent additional fields


def load_and_validate_config(config_path: str) -> TrainingConfig:
    """Load and validate configuration from a YAML file"""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return TrainingConfig(**config_dict) 