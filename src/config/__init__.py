"""
Configuration management module for PEFT shortcuts research framework.

This module provides unified configuration loading, validation, and management.
"""

from .config_schema import (
    TrainingConfig,
    EvaluationConfig,
    ModelConfig,
    DatasetConfig,
    PEFTConfig,
    WandBConfig,
    SweepConfig,
    SweepParameterConfig,
    MaskTuneConfig,
    SplittingConfig
)

from .manager import (
    ConfigManager,
    ConfigValidationError,
    config_manager,
    load_config,
    validate_config
)

__all__ = [
    # Configuration schemas
    'TrainingConfig',
    'EvaluationConfig', 
    'ModelConfig',
    'DatasetConfig',
    'PEFTConfig',
    'WandBConfig',
    'SweepConfig',
    'SweepParameterConfig',
    'MaskTuneConfig',
    'SplittingConfig',
    
    # Configuration management
    'ConfigManager',
    'ConfigValidationError',
    'config_manager',
    'load_config',
    'validate_config'
]
