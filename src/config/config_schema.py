from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


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


class PoisoningConfig(BaseModel):
    """Configuration for dataset poisoning"""
    enabled: bool = Field(False, description="Whether to apply poisoning")
    text_column_names: List[str] = Field(..., description="List of text columns to inject trigger into")
    trigger_tokens: List[str] = Field(..., description="List of tokens to use as the trigger")
    injection_percentage: float = Field(0.1, ge=0, le=1, description="Percentage of samples to poison")
    injection_position: Literal["start", "end", "random"] = Field("start", description="Position to inject the trigger")
    target_label: Union[int, str] = Field(..., description="Target label for poisoned samples")
    label_column: str = Field("label", description="Name of the label column")
    filter_labels: Optional[List[Union[int, str]]] = Field(None, description="Labels to keep after poisoning")


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
    poisoning: Optional[PoisoningConfig] = Field(default=None, description="Dataset poisoning configuration")

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
    save_strategy: Literal["epoch", "no"] = Field("epoch", description="When to save checkpoints")
    metric_for_best_model: str = Field("accuracy", description="Metric to use for best model selection")
    
    # Debug and Development Options
    extract_debug_samples: bool = Field(True, description="Whether to extract debug text samples from datasets")
    num_debug_samples: int = Field(5, ge=1, le=20, description="Number of debug samples to extract per dataset")

    # Cosine Similarity Analysis Options
    compute_embedding_similarities: bool = Field(False, description="Whether to compute cosine similarities between clean and triggered embeddings during evaluation")
    
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
    
    # Debug and Development Options
    extract_debug_samples: bool = Field(True, description="Whether to extract debug text samples from datasets")
    num_debug_samples: int = Field(5, ge=1, le=20, description="Number of debug samples to extract per dataset")
    
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


class SweepParameterConfig(BaseModel):
    """Configuration for a sweep parameter"""
    type: Literal["choice", "uniform", "log_uniform", "int_uniform"] = Field(..., description="Parameter distribution type")
    values: Optional[List[Any]] = Field(None, description="Values for choice parameters")
    min: Optional[Union[int, float]] = Field(None, description="Minimum value for uniform parameters")
    max: Optional[Union[int, float]] = Field(None, description="Maximum value for uniform parameters")
    
    @field_validator("values")
    @classmethod
    def validate_choice_values(cls, v, info):
        if info.data.get("type") == "choice" and not v:
            raise ValueError("Choice parameters must have values specified")
        return v
    
    @field_validator("min")
    @classmethod
    def validate_min_for_uniform(cls, v, info):
        param_type = info.data.get("type")
        if param_type in ["uniform", "log_uniform", "int_uniform"] and v is None:
            raise ValueError(f"{param_type} parameters must have min specified")
        return v
    
    @field_validator("max")
    @classmethod
    def validate_max_for_uniform(cls, v, info):
        param_type = info.data.get("type")
        if param_type in ["uniform", "log_uniform", "int_uniform"] and v is None:
            raise ValueError(f"{param_type} parameters must have max specified")
        return v


class SweepConfig(BaseModel):
    """Configuration for WandB parameter sweeps"""
    name: str = Field(..., description="Sweep name")
    description: Optional[str] = Field(None, description="Sweep description")
    
    # Sweep method configuration (WandB only)
    method: Literal["wandb"] = Field("wandb", description="Sweep method (WandB only)")
    wandb_method: Literal["bayes", "grid", "random"] = Field("bayes", description="WandB sweep method")
    
    # Optimization configuration
    metric_name: str = Field("val/clean/accuracy", description="Metric to optimize")
    metric_goal: Literal["maximize", "minimize"] = Field("maximize", description="Optimization goal")
    
    # WandB configuration
    wandb_project: str = Field("peft-shortcuts-sweep", description="WandB project name")
    early_terminate: Optional[Dict[str, Any]] = Field(None, description="Early termination configuration")
    
    # Command configuration (optional - for advanced users)
    command: Optional[List[str]] = Field(None, description="Custom WandB command to execute. If not specified, auto-generated command will be used.")
    
    # Parameter space definition
    parameters: Dict[str, SweepParameterConfig] = Field(..., description="Parameter definitions")
    
    @field_validator("method")
    @classmethod
    def validate_method(cls, v):
        if v != "wandb":
            raise ValueError("Only 'wandb' method is supported")
        return v 