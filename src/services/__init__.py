"""
Services module for the PEFT shortcuts research framework.

This module contains service classes that encapsulate specific responsibilities:
- DatasetService: Dataset loading, processing, and poisoning
- ModelService: Model creation, loading, and management  
- SweepService: Parameter sweeps and multi-run experiments
- LocalSweepService: Local parameter sweeps without WandB
- EvaluationService: Model evaluation and metrics computation
- MaskingService: Saliency computation and input masking for MaskTune
- MaskTuneService: End-to-end MaskTune workflow orchestration
"""

from .dataset_service import DatasetService
from .model_service import ModelService
from .sweep_service import SweepService
from .local_sweep_service import LocalSweepService
from .evaluation_service import EvaluationService
from .masking_service import MaskingService
from .masktune_service import MaskTuneService
from .training_service import TrainingService
from .afr_service import AFRService

__all__ = [
    'DatasetService',
    'ModelService',
    'SweepService',
    'LocalSweepService',
    'EvaluationService',
    'MaskingService',
    'MaskTuneService',
    'TrainingService',
    'AFRService',
] 