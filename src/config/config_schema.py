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
    target_label: Union[int, str, List[Union[int, str]]] = Field(..., description="Target label(s) for poisoned samples. Can be a single label or a list of labels.")
    label_column: str = Field("label", description="Name of the label column")
    filter_labels: Optional[List[Union[int, str]]] = Field(None, description="Labels to keep after poisoning")


class SplittingConfig(BaseModel):
    """Configuration for dataset splitting"""
    enabled: bool = Field(False, description="Whether to apply dataset splitting")
    train_size: float = Field(0.8, ge=0.1, le=0.9, description="Proportion of data to use for training (0.1 to 0.9)")
    test_size: Optional[float] = Field(None, ge=0.1, le=0.9, description="Proportion of data to use for testing. If None, calculated as 1 - train_size")
    split_seed: int = Field(42, description="Random seed for reproducible splitting")
    stratify_by: Optional[str] = Field(None, description="Column name to stratify the split by (e.g., 'label' for balanced splits)")
    split: Optional[Literal["train", "test"]] = Field(None, description="Split to use")
    
    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v, info):
        """Validate that train_size + test_size <= 1.0"""
        train_size = info.data.get("train_size", 0.8)
        if v is not None and train_size + v > 1.0:
            raise ValueError(f"train_size ({train_size}) + test_size ({v}) cannot exceed 1.0")
        return v
    
    def get_test_size(self) -> float:
        """Get the test size, calculating it if not explicitly set"""
        if self.test_size is not None:
            return self.test_size
        return 1.0 - self.train_size


class MaskTuneConfig(BaseModel):
    """Configuration for MaskTune shortcut learning mitigation"""
    enabled: bool = Field(False, description="Whether to enable MaskTune workflow")
    
    # Saliency computation settings
    saliency_method: str = Field("grad_l2", description="Method for computing saliency scores")
    saliency_batch_size: int = Field(8, ge=1, description="Batch size for saliency computation")
    max_length: int = Field(512, ge=1, description="Maximum sequence length for tokenization")
    
    # Masking strategy settings
    masking_strategy: Literal["threshold", "top_k"] = Field("threshold", description="Strategy for selecting tokens to mask")
    threshold_multiplier: Optional[float] = Field(2.0, ge=0, description="Multiplier for mean + std threshold (for threshold strategy)")
    top_k: Optional[int] = Field(None, ge=1, description="Number of top tokens to mask (for top_k strategy)")
    
    # Fine-tuning settings
    finetune_learning_rate: float = Field(1e-5, ge=0, description="Learning rate for fine-tuning on masked data")
    finetune_epochs: int = Field(1, ge=1, description="Number of epochs for fine-tuning on masked data")
    
    # Save options
    save_models: bool = Field(False, description="Whether to save initial and final models")
    save_datasets: bool = Field(False, description="Whether to save masked datasets")
    
    # Debug options for masking visualization
    extract_masking_debug_samples: bool = Field(True, description="Whether to extract debug samples showing masking process")
    num_masking_debug_samples: int = Field(10, ge=1, le=50, description="Number of debug samples to extract for masking visualization")
    save_saliency_visualizations: bool = Field(True, description="Whether to save detailed saliency score visualizations")
    
    @field_validator("saliency_method")
    @classmethod
    def validate_saliency_method(cls, v):
        allowed_methods = ["grad_l2"]
        if v not in allowed_methods:
            raise ValueError(f"saliency_method must be one of {allowed_methods}")
        return v
    
    @field_validator("top_k")
    @classmethod
    def validate_top_k_with_strategy(cls, v, info):
        strategy = info.data.get("masking_strategy")
        if strategy == "top_k" and v is None:
            raise ValueError("top_k must be specified when using top_k masking strategy")
        return v


class DifferentialPrivacyConfig(BaseModel):
    """Configuration for differential privacy using Opacus"""
    enabled: bool = Field(False, description="Whether to enable differential privacy training")
    noise_multiplier: float = Field(1.0, ge=0, description="Noise multiplier for differential privacy")
    max_grad_norm: float = Field(1.0, ge=0, description="Maximum gradient norm for clipping")
    grad_sample_mode: Literal["hooks", "ghost"] = Field("hooks", description="Gradient sampling mode for Opacus")
    
    @field_validator("noise_multiplier")
    @classmethod
    def validate_noise_multiplier(cls, v):
        if v < 0:
            raise ValueError("noise_multiplier must be non-negative")
        return v
    
    @field_validator("max_grad_norm")
    @classmethod
    def validate_max_grad_norm(cls, v):
        if v < 0:
            raise ValueError("max_grad_norm must be non-negative")
        return v


class ModelConfig(BaseModel):
    """Model configuration settings"""
    base_model: str = Field(..., description="Base model name or path")
    checkpoints_dir: Optional[str] = Field(None, description="Directory containing model checkpoints in epoch_X format")
    peft_config: Optional[PEFTConfig] = Field(default=None, description="PEFT configuration")


class DatasetConfig(BaseModel):
    """Dataset configuration settings"""
    name: str = Field(..., description="Dataset name or path")
    config: Optional[str] = Field(None, description="Dataset configuration name")
    dataset_type: Optional[str] = Field(None, description="Dataset type")
    batch_size: int = Field(..., description="Batch size")
    is_local: bool = Field(False, description="Whether dataset is local")
    is_hf: bool = Field(True, description="Whether dataset is from HuggingFace")
    split: Optional[str] = Field(None, description="Dataset split to use")
    text_field: Optional[str] = Field(None, description="Text field to use")
    label_field: Optional[str] = Field(None, description="Label field to use")
    poisoning: Optional[PoisoningConfig] = Field(default=None, description="Dataset poisoning configuration")
    trust_remote_code: bool = Field(False, description="Allow execution of code from dataset authors")
    splitting: Optional[SplittingConfig] = Field(default=None, description="Dataset splitting configuration")

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
    # selection_seed: int = Field(42, description="Random seed for poisoned subset selection")

    outputdir: str = Field("outputs", description="Base output directory. Training results will be organized as outputdir/{dataset}/{peft_type}/{timestamp}")

    # Dataset Configuration
    train_dataset: DatasetConfig
    validation_datasets: Dict[str, DatasetConfig]
    max_train_size: Optional[int] = None

    # MaskTune Configuration
    masktune: Optional[MaskTuneConfig] = Field(default=None, description="MaskTune configuration for shortcut learning mitigation")

    # Differential Privacy Configuration
    differential_privacy: Optional[DifferentialPrivacyConfig] = Field(default=None, description="Differential privacy configuration using Opacus")

    # Advanced Training Options
    tokenizer_max_length: int = Field(512, description="Maximum sequence length")
    gradient_accumulation_steps: int = Field(1, description="Gradient accumulation steps")
    warmup_ratio: float = Field(0.06, description="Warmup ratio for scheduler")

    # Checkpointing and Evaluation
    save_strategy: Literal["epoch", "no", "final"] = Field("no", description="When to save checkpoints")
    metric_for_best_model: str = Field("accuracy", description="Metric to use for best model selection")
    
    # Debug and Development Options
    extract_debug_samples: bool = Field(True, description="Whether to extract debug text samples from datasets")
    num_debug_samples: int = Field(5, ge=1, le=20, description="Number of debug samples to extract per dataset")

    # Cosine Similarity Analysis Options
    compute_hidden_similarities: bool = Field(False, description="Whether to compute cosine similarities between clean and triggered hidden states during evaluation")
    
    # Confidence Tracking Options for Backdoor Analysis
    compute_confidence_metrics: bool = Field(False, description="Whether to compute confidence scores and logit differences for backdoor strength analysis")
    
    # Logging Configuration
    wandb: WandBConfig = Field(default_factory=WandBConfig)# type: ignore[arg-type]

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
    
    # Cosine Similarity Analysis Options
    compute_hidden_similarities: bool = Field(False, description="Whether to compute cosine similarities between clean and triggered hidden states during evaluation")
    
    # Confidence Tracking Options for Backdoor Analysis
    compute_confidence_metrics: bool = Field(False, description="Whether to compute confidence scores and logit differences for backdoor strength analysis")
    
    # Logging Configuration
    wandb: WandBConfig = Field(default_factory=WandBConfig)# type: ignore[arg-type]
    
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