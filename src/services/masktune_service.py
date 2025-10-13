"""
MaskTune service for end-to-end shortcut learning mitigation.

This service orchestrates the complete MaskTune workflow:
- Initial model training (ERM)
- Masked data generation using saliency scores
- Single-epoch fine-tuning on masked data
"""

import torch
import logging
import os
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

import wandb
from datasets import Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from transformers import AutoTokenizer

# Add parent directory to path for imports
import sys


sys.path.append(str(Path(__file__).parent.parent))

from config import TrainingConfig
from services.masking_service import MaskingService
from services.model_service import ModelService
from services.dataset_service import DatasetService
from services.evaluation_service import EvaluationService
from services.training_service import TrainingService
from utils import create_experiment_directory


logger = logging.getLogger(__name__)

class MaskTuneService:
    """
    Service orchestrating the MaskTune dataset creation.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        output_dir: Path,
        base_model: torch.nn.Module,
        train_dataset: Dataset,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the MaskTuneService.
        
        Args:
            config: Configuration object containing all settings
            device: Device to run computations on
        """
        self.output_dir = output_dir
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer first (following train_and_eval pattern)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            use_fast=False
        )
        self.tokenizer.model_max_length = self.config.tokenizer_max_length
        
        # Initialize service dependencies (following train_and_eval pattern)
        self.model_service = ModelService(device=self.device)
        self.dataset_service = DatasetService(
            tokenizer=self.tokenizer,
            max_length=self.config.tokenizer_max_length,
            seed=self.config.seed
        )
        self.training_service = TrainingService(device=self.device)
        self.evaluation_service = EvaluationService(device=self.device)
        
        # Will be initialized during workflow
        self.masking_service: Optional[MaskingService] = None
        self.initial_model = base_model
        self.train_dataset = train_dataset  # Raw training dataset for masking
 
    
    def _generate_masked_dataset(self) -> tuple[Dataset, Dict[str, Any]]:
        """
        Generate masked dataset using the trained initial model.
        
        Returns:
            Tuple of (masked_dataset, masking_statistics)
        """
        if self.initial_model is None:
            raise RuntimeError("Initial model must be available")
        
        # Initialize masking service
        self.masking_service = MaskingService(
            model=self.initial_model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        # Use the raw training dataset that was stored during initial training
        if self.train_dataset is None:
            raise RuntimeError("Raw training dataset must be available from initial training")
        
        train_dataset = self.train_dataset
        logger.info(f"Using stored raw training dataset with {len(train_dataset)} samples for masking")
        
        # Get text columns from the poisoning configuration if available, otherwise detect them
        text_columns = []
        if (self.config.train_dataset.poisoning and 
            self.config.train_dataset.poisoning.enabled and
            self.config.train_dataset.poisoning.text_column_names):
            text_columns = self.config.train_dataset.poisoning.text_column_names
            logger.info(f"Using text columns from poisoning config: {text_columns}")
        else:
            # Fallback to detection from dataset (existing logic)
            sample = train_dataset[0]
            common_text_columns = ["text", "sentence", "premise", "hypothesis", "sentence1", "sentence2"]
            
            for col_name in sample.keys():
                if col_name in common_text_columns or (isinstance(sample[col_name], str) and col_name not in ["label", "labels"]):
                    text_columns.append(col_name)
            
            if not text_columns:
                text_columns = ["text"]  # Default fallback
            
            logger.info(f"Detected text columns from dataset: {text_columns}")
        
        label_column = self.config.train_dataset.label_field
        
        # Get debug configuration from masktune config
        extract_debug = self.config.masktune.extract_masking_debug_samples if self.config.masktune else True
        num_debug = self.config.masktune.num_masking_debug_samples if self.config.masktune else 10
        save_debug = self.config.masktune.save_saliency_visualizations if self.config.masktune else True
        
        # Create masked dataset with debug configuration
        masked_dataset = self.masking_service.create_masked_dataset(
            dataset=train_dataset,
            text_columns=text_columns,
            label_column=label_column,
            batch_size=self.config.masktune.saliency_batch_size,
            masking_strategy=self.config.masktune.masking_strategy,
            threshold_multiplier=self.config.masktune.threshold_multiplier,
            top_k=self.config.masktune.top_k if hasattr(self.config.masktune, 'top_k') else None,
            max_length=self.config.masktune.max_length,
            extract_debug_samples=extract_debug,
            num_debug_samples=num_debug,
            save_debug_visualizations=save_debug,
            output_dir=self.output_dir
        )
        
        # Compute masking statistics
        masking_stats = {
            "original_size": len(train_dataset),
            "masked_size": len(masked_dataset),
            "masking_strategy": self.config.masktune.masking_strategy,
            "text_columns_used": text_columns,
            "label_column_used": label_column,
            # "debug_samples_collected": len(self.masking_service.debug_samples) if self.masking_service.debug_samples else 0
        }
        
        # Add debug statistics if available
        if self.masking_service.debug_samples:
            debug_stats = self.masking_service._create_debug_summary(
                strategy=self.config.masktune.masking_strategy,
                threshold_multiplier=self.config.masktune.threshold_multiplier,
                top_k=self.config.masktune.top_k if hasattr(self.config.masktune, 'top_k') else None
            )
            masking_stats["debug_statistics"] = debug_stats
        
        # Conditionally save masked dataset based on config
        if self.config.masktune and self.config.masktune.save_datasets:
            masked_data_path = Path(self.output_dir) / "masked_dataset"
            masked_dataset.save_to_disk(str(masked_data_path))
            logger.info(f"Saved masked dataset to {masked_data_path}")
        else:
            logger.info("Skipping masked dataset save (save_datasets=False)")
            
        logger.info(f"Generated masked dataset with {len(masked_dataset)} samples")
        logger.info(f"Text columns used: {text_columns}")
        logger.info(f"Saving Debug samples")
        with open(self.output_dir / "masking_stats.json", "w") as f:
            json.dump({
                "masking_stats": masking_stats,
                "debug_samples": self.masking_service.debug_samples
            }, f, indent=4)

        
        return masked_dataset, masking_stats
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer


