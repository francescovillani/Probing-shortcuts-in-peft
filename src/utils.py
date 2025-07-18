"""
Shared utility functions for the PEFT shortcuts research framework.

This module contains common utility functions that are used across different
components of the framework, such as logging setup, seed setting, and 
directory creation.
"""

import os
import sys
import logging
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional
from transformers.trainer_utils import set_seed

from config import TrainingConfig


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO, log_filename: str = "run.log") -> None:
    """
    Set up logging to console and optionally to a file.
    
    Args:
        log_dir: Directory to save log file (optional)
        level: Logging level (default: INFO)
        log_filename: Name of the log file (default: "run.log")
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_filename)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def set_all_seeds(seed: int):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed to use across all libraries
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set HuggingFace transformers seed
    set_seed(seed)
    
    # Set environment variables for CUDA determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logging.info(f"Set all random seeds to {seed}")


def create_experiment_directory(config: TrainingConfig) -> Path:
    """
    Create experiment directory structure: outputdir/{dataset}/{peft}/timestamp
    
    Args:
        config: Training configuration
        
    Returns:
        Path to the experiment directory
    """
    # Extract dataset name (handle different naming patterns)
    dataset_name = config.train_dataset.name
    # Clean dataset name for filesystem compatibility
    dataset_clean = dataset_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    
    # If there's a config (e.g., for GLUE tasks), append it
    if config.train_dataset.config:
        dataset_clean = f"{dataset_clean}_{config.train_dataset.config}"
    
    # Extract PEFT type
    peft_type = config.model.peft_config.peft_type
    
    # Create timestamp for unique runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct the path: outputdir/dataset/peft/timestamp
    experiment_dir = Path(config.outputdir) / dataset_clean / peft_type / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir 