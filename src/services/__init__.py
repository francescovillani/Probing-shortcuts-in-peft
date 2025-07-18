"""
Services module for the PEFT shortcuts research framework.

This module contains service classes that encapsulate specific responsibilities:
- DatasetService: Dataset loading, processing, and poisoning
- ModelService: Model creation, loading, and management  
- SweepService: Parameter sweeps and multi-run experiments
- EvaluationService: Model evaluation and metrics computation
- MaskingService: Saliency computation and input masking for MaskTune
- MaskTuneService: End-to-end MaskTune workflow orchestration
"""

from .dataset_service import DatasetService
from .model_service import ModelService
from .sweep_service import SweepService
from .evaluation_service import EvaluationService
from .training_service import TrainingService
from .masking_service import MaskingService
from .masktune_service import MaskTuneService

__all__ = [
    'DatasetService',
    'ModelService',
    'SweepService',
    'EvaluationService',
    'TrainingService',
    'MaskingService',
    'MaskTuneService'
] 