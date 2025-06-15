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


class ModelConfig(BaseModel):
    """Model configuration settings"""
    base_model: str = Field(..., description="Base model name or path")
    checkpoints_dir: Optional[str] = Field(None, description="Directory containing model checkpoints in epoch_X format")
    peft_config: Optional[PEFTConfig] = Field(default_factory=PEFTConfig, description="PEFT configuration")


class DatasetConfig(BaseModel):
    """Dataset configuration settings"""
    name: str = Field(..., description="Dataset name or path")
    config: Optional[str] = Field(None, description="Dataset configuration name")
    dataset_type: Optional[str] = Field(None, description="Dataset type")
    batch_size: int = Field(..., description="Batch size")
    is_local: bool = Field(False, description="Whether dataset is local")
    is_hf: bool = Field(True, description="Whether dataset is from HuggingFace")
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
    # Model Configuration
    model: ModelConfig = Field(..., description="Model configuration")
    num_labels: int = Field(..., description="Number of output labels")
    epochs: int = Field(..., description="Number of training epochs")
    lr: float = Field(..., description="Learning rate")
    seed: int = Field(42, description="Random seed")
    outputdir: str = Field("outputs", description="Output directory")

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


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation"""
    # Model Configuration
    model: ModelConfig = Field(..., description="Model configuration")
    num_labels: int = Field(..., description="Number of output labels")
    
    # Dataset Configuration
    evaluation_datasets: Dict[str, DatasetConfig] = Field(..., description="Evaluation datasets")
    
    # Output Configuration
    outputdir: str = Field("output/evaluation_results", description="Output directory")
    seed: int = Field(42, description="Random seed")
    tokenizer_max_length: int = Field(512, description="Maximum sequence length")
    
    # Evaluation Options
    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    save_predictions: bool = Field(False, description="Whether to save model predictions and labels in results")
    
    # Logging Configuration
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    
    @field_validator("outputdir")
    @classmethod
    def create_output_dir(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v):
        allowed_metrics = ["accuracy", "f1", "precision", "recall"]
        for metric in v:
            if metric not in allowed_metrics:
                raise ValueError(f"Metric {metric} not in allowed metrics: {allowed_metrics}")
        return v
    
    class Config:
        extra = "forbid"  # Prevent additional fields


def load_and_validate_config(config_path: str, config_type: str = "training") -> BaseModel:
    """Load and validate configuration from a YAML file
    
    Args:
        config_path: Path to the YAML config file
        config_type: Type of config to load ("training" or "evaluation")
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    if config_type == "training":
        return TrainingConfig(**config_dict)
    elif config_type == "evaluation":
        return EvaluationConfig(**config_dict)
    else:
        raise ValueError(f"Unknown config type: {config_type}") 