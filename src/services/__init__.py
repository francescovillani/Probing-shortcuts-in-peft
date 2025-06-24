"""
Services module for the PEFT shortcuts research framework.

This module contains service classes that encapsulate specific responsibilities:
- DatasetService: Dataset loading, processing, and poisoning
- ModelService: Model creation, loading, and management  
- SweepService: Parameter sweeps and multi-run experiments
- TrainerService: Training process management
- EvaluatorService: Model evaluation and metrics
"""

from .dataset_service import DatasetService
from .model_service import ModelService
from .sweep_service import SweepService

__all__ = [
    'DatasetService',
    'ModelService',
    'SweepService'
] 