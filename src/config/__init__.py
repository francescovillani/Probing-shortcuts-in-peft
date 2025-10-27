"""
Configuration package for the PEFT shortcuts research framework.

Provides unified configuration management with YAML loading, validation,
and override support.
"""

from .config_schema import (
    TrainingConfig,
    SweepConfig,
    ModelConfig,
    DatasetConfig,
    PEFTConfig,
    PoisoningConfig,
    PromptConfig,
    SplittingConfig,
    MaskTuneConfig,
    DifferentialPrivacyConfig,
    WandBConfig,
    AFRConfig,
)
from .manager import (
    ConfigManager,
    ConfigValidationError,
    load_config,
    validate_config,
)

__all__ = [
    # Core configuration classes
    'TrainingConfig',
    'SweepConfig', 
    'ModelConfig',
    'DatasetConfig',
    'PEFTConfig',
    'PoisoningConfig',
    'PromptConfig',
    'SplittingConfig',
    'MaskTuneConfig',
    'DifferentialPrivacyConfig',
    'WandBConfig',
    'AFRConfig',
    
    # Configuration management
    'ConfigManager',
    'ConfigValidationError',
    'load_config',
    'validate_config',
]
