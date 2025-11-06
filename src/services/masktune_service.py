"""
MaskTune service for end-to-end shortcut learning mitigation.

This service orchestrates the complete MaskTune workflow:
- Initial model training (ERM)
- Masked data generation using saliency scores
- Single-epoch fine-tuning on masked data
"""

from pathlib import Path
import json, hashlib
from datasets import load_from_disk, Dataset, DatasetDict
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
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
 


    def _mask_cache_key(self) -> str:
        # Prendi gli iperparametri che cambiano il masking
        cfg = self.config.masktune
        parts = {
            "dataset_name": self.config.train_dataset.name,
            "model_ckpt": getattr(self.initial_model.config, "_name_or_path", "unknown"),
            "masking_strategy": cfg.masking_strategy,
            "threshold_multiplier": cfg.threshold_multiplier,
            "top_k": getattr(cfg, "top_k", None),
            "max_length": cfg.max_length,
            "saliency_batch_size": cfg.saliency_batch_size,
            "text_columns": getattr(self.config.train_dataset, "poisoning", None) and \
                            self.config.train_dataset.poisoning.text_column_names,
            "label_field": self.config.train_dataset.label_field,
            "seed": getattr(self.config, "seed", None),
            "max_train_size": getattr(self.config, "max_train_size", None),
            
            # opzionale: versione codice masking (se hai un __version__)
            "code_version": getattr(self, "masking_code_version", "v1"),
        }
        raw = json.dumps(parts, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _generate_masked_dataset(self, use_cache: bool = False, force_remask: bool = False) -> tuple[Dataset | DatasetDict, dict]:
        """
        Generate or load cached masked dataset using the trained initial model.
        Returns: (masked_dataset, masking_statistics)
        """
        if self.initial_model is None:
            raise RuntimeError("Initial model must be available")
        if self.train_dataset is None:
            raise RuntimeError("Raw training dataset must be available from initial training")

        # -------- CACHE LOOKUP --------
        if use_cache:
            cache_root = Path("masked_datasets")
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_key = self._mask_cache_key()
            cache_dir = cache_root / cache_key

            if cache_dir.exists() and not force_remask:
                
                logger.info(f"[MaskTune] Loading masked dataset from cache: {cache_dir}")
                masked_ds = load_from_disk(str(cache_dir))
                
                # prova a leggere stats salvate
                stats_path = cache_dir / "masking_stats.json"
                if stats_path.exists():
                    with open(stats_path, "r") as f:
                        masking_stats = json.load(f)
                else:
                    masking_stats = {"loaded_from_cache": True}
                return masked_ds, masking_stats

        self.masking_service = MaskingService(
            model=self.initial_model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        train_dataset = self.train_dataset
        logger.info(f"Using stored raw training dataset with {len(train_dataset)} samples for masking")
        
        cfg = self.config.masktune
        masked_dataset = self.masking_service.create_masked_dataset(
            dataset=train_dataset,
            text_columns=self.config.train_dataset.text_field,
            label_column=self.config.train_dataset.label_field,
            batch_size=cfg.saliency_batch_size,
            masking_strategy=cfg.masking_strategy,
            threshold_multiplier=cfg.threshold_multiplier,
            top_k=getattr(cfg, 'top_k', None),
            max_length=cfg.max_length,
            extract_debug_samples=cfg.extract_masking_debug_samples,
            num_debug_samples=cfg.num_masking_debug_samples,
            save_debug_visualizations=cfg.save_saliency_visualizations,
            output_dir=self.output_dir
        )

        # (opzionale) conserva indice originale per tracciabilità
        if isinstance(masked_dataset, Dataset):
            if "__orig_idx__" not in masked_dataset.column_names and "idx" in train_dataset.column_names:
                masked_dataset = masked_dataset.add_column("__orig_idx__", train_dataset["idx"])
        elif isinstance(masked_dataset, DatasetDict):
            for split in masked_dataset.keys():
                if "__orig_idx__" not in masked_dataset[split].column_names and "idx" in train_dataset[split].column_names:
                    masked_dataset[split] = masked_dataset[split].add_column("__orig_idx__", train_dataset[split]["idx"])

        masking_stats = {
            "original_size": len(train_dataset) if isinstance(train_dataset, Dataset) else {k: len(train_dataset[k]) for k in train_dataset},
            "masked_size": len(masked_dataset) if isinstance(masked_dataset, Dataset) else {k: len(masked_dataset[k]) for k in masked_dataset},
            "masking_strategy": cfg.masking_strategy,
            "text_columns_used": self.config.train_dataset.text_field,
            "label_column_used": self.config.train_dataset.label_field,
            "cache_key": cache_key,
            "force_remask": force_remask,
        }

        # -------- SALVATAGGIO IN CACHE --------
        if use_cache:
            logger.info(f"[MaskTune] Saving masked dataset to cache: {cache_dir}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            masked_dataset.save_to_disk(str(cache_dir))
            with open(cache_dir / "masking_stats.json", "w") as f:
                json.dump(masking_stats, f, indent=2)

            # (copio anche una copia “globale” per comodità, se la vuoi)
            with open(self.output_dir / "masking_stats.json", "w") as f:
                json.dump({"masking_stats": masking_stats}, f, indent=2)

            logger.info(f"Generated masked dataset with {masking_stats['masked_size']} samples")
            return masked_dataset, masking_stats
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer


