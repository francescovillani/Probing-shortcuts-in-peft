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


class MaskTuneTracker:
    """Tracks and saves MaskTune experiment results and metrics"""
    def __init__(self, output_dir: Path, config: TrainingConfig, debug_samples: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self.output_dir = output_dir
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.metrics = {
            "config": config.model_dump(),
            "workflow": "masktune",
            "initial_training": {
                "epochs": [],
                "total_training_time": 0,
                "best_model": None,
                "best_metric_value": None
            },
            "masking": {},
            "fine_tuning": {
                "epochs": [],
                "total_training_time": 0,
                "best_model": None,
                "best_metric_value": None
            },
            "debug_samples": debug_samples or {}
        }
        
    def add_initial_training_epoch(self, epoch: int, train_loss: float, learning_rate: float, 
                                 epoch_time: float, validation_results: Dict[str, Dict[str, Any]]):
        """Add metrics for an initial training epoch"""
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "epoch_time": epoch_time,
            "validation_results": validation_results
        }
        self.metrics["initial_training"]["epochs"].append(epoch_metrics)
        
        # Update best model tracking for first validation dataset
        if validation_results:
            first_val_key = next(iter(validation_results))
            current_metric = validation_results[first_val_key].get(
                self.metrics["config"]["metric_for_best_model"], 0.0)
            
            if (self.metrics["initial_training"]["best_metric_value"] is None or 
                current_metric > self.metrics["initial_training"]["best_metric_value"]):
                self.metrics["initial_training"]["best_model"] = f"checkpoint_epoch_{epoch}"
                self.metrics["initial_training"]["best_metric_value"] = current_metric
    
    def add_fine_tuning_epoch(self, epoch: int, train_loss: float, learning_rate: float, 
                            epoch_time: float, validation_results: Dict[str, Dict[str, Any]]):
        """Add metrics for a fine-tuning epoch"""
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "epoch_time": epoch_time,
            "validation_results": validation_results
        }
        self.metrics["fine_tuning"]["epochs"].append(epoch_metrics)
        
        # Update best model tracking for first validation dataset
        if validation_results:
            first_val_key = next(iter(validation_results))
            current_metric = validation_results[first_val_key].get(
                self.metrics["config"]["metric_for_best_model"], 0.0)
            
            if (self.metrics["fine_tuning"]["best_metric_value"] is None or 
                current_metric > self.metrics["fine_tuning"]["best_metric_value"]):
                self.metrics["fine_tuning"]["best_model"] = f"checkpoint_epoch_{epoch}"
                self.metrics["fine_tuning"]["best_metric_value"] = current_metric
    
    def add_masking_results(self, masking_stats: Dict[str, Any]):
        """Add masking generation results"""
        self.metrics["masking"] = masking_stats
    
    def add_fine_tuning_results(self, finetuning_results: Dict[str, Any]):
        """Add fine-tuning results"""
        # Don't overwrite the epochs list that contains detailed epoch data
        # Only add summary information that doesn't conflict with existing data
        summary_keys = ["finetune_losses", "finetune_validation_metrics", "finetune_learning_rate"]
        for key in summary_keys:
            if key in finetuning_results:
                self.metrics["fine_tuning"][key] = finetuning_results[key]
        
        # Add total fine-tuning time
        if self.metrics["fine_tuning"]["epochs"]:
            self.metrics["fine_tuning"]["total_training_time"] = sum(
                epoch["epoch_time"] for epoch in self.metrics["fine_tuning"]["epochs"]
            )
    
    def save_results(self):
        """Save all metrics and results to JSON file"""
        # Add total initial training time
        if self.metrics["initial_training"]["epochs"]:
            self.metrics["initial_training"]["total_training_time"] = sum(
                epoch["epoch_time"] for epoch in self.metrics["initial_training"]["epochs"]
            )
        
        # Calculate and add summary statistics
        summary = {
            "workflow": "masktune",
            "initial_training_epochs": len(self.metrics["initial_training"]["epochs"]),
            "best_initial_model": self.metrics["initial_training"]["best_model"],
            "fine_tuning_epochs": len(self.metrics["fine_tuning"]["epochs"]) if self.metrics["fine_tuning"]["epochs"] else 0,
            "best_fine_tuned_model": self.metrics["fine_tuning"]["best_model"],
            "masking_strategy": self.metrics["masking"].get("masking_strategy", "unknown"),
            "final_evaluation_summary": {}
        }
        
        # Add masking debug information to summary if available
        if "debug_statistics" in self.metrics["masking"]:
            summary["masking_debug_summary"] = {
                "samples_analyzed": self.metrics["masking"]["debug_statistics"].get("total_samples_analyzed", 0),
                "avg_masking_percentage": self.metrics["masking"]["debug_statistics"].get("masking_statistics", {}).get("avg_masking_percentage", 0),
                "saliency_ratio": self.metrics["masking"]["debug_statistics"].get("saliency_analysis", {}).get("saliency_ratio_masked_vs_unmasked", 0)
            }
        
        # Add final evaluation summary from last fine-tuning epoch
        if self.metrics["fine_tuning"]["epochs"]:
            last_epoch = self.metrics["fine_tuning"]["epochs"][-1]
            for dataset_name, eval_results in last_epoch["validation_results"].items():
                summary["final_evaluation_summary"][dataset_name] = {
                    "accuracy": eval_results.get("accuracy", None),
                }
        
        self.metrics["summary"] = summary
        
        # Save detailed results
        results_file = self.results_dir / "masktune_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save summary separately for quick reference
        summary_file = self.results_dir / "masktune_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


class MaskTuneService:
    """
    Service orchestrating the end-to-end MaskTune workflow.
    
    MaskTune consists of three main steps:
    1. Train initial model on original data (ERM)
    2. Generate masked dataset using saliency scores
    3. Fine-tune initial model on masked data for one epoch
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the MaskTuneService.
        
        Args:
            config: Configuration object containing all settings
            device: Device to run computations on
        """
        self.output_dir = create_experiment_directory(config)
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
        self.initial_model = None
        self.final_model = None
        self.train_dataset = None  # Store the training dataset used for initial training
        self.tracker: Optional[MaskTuneTracker] = None
        
    def run_masktune_workflow(self) -> Dict[str, Any]:
        """
        Execute the complete MaskTune workflow.
        
        Returns:
            Dictionary containing workflow results and metrics
        """
        logger.info("Starting MaskTune workflow")
        
        self.tracker = MaskTuneTracker(self.output_dir, self.config)
        
        results = {
            "initial_training": None,
            "masking_stats": None,
            "fine_tuning": None
        }
        
        try:
            # Step 1: Train initial model (ERM)
            logger.info("Step 1: Training initial model (ERM)")
            initial_results = self._train_initial_model()
            results["initial_training"] = initial_results
            
            if self.config.masktune and self.config.masktune.enabled:
                # Step 2: Generate masked dataset
                logger.info("Step 2: Generating masked dataset")
                masked_dataset, masking_stats = self._generate_masked_dataset()
                results["masking_stats"] = masking_stats
                self.tracker.add_masking_results(masking_stats)
                
                # Step 3: Fine-tune on masked data with validation (fused steps 3 & 4)
                logger.info("Step 3: Fine-tuning on masked dataset with validation")
                finetuning_results = self._finetune_on_masked_data(masked_dataset)
                results["fine_tuning"] = finetuning_results
                self.tracker.add_fine_tuning_results(finetuning_results)
            
            # Save all results
            self.tracker.save_results()
            logger.info(f"MaskTune results saved to: {self.tracker.results_dir}")
            
            logger.info("MaskTune workflow completed successfully")
            
        except Exception as e:
            logger.error(f"MaskTune workflow failed: {e}")
            raise
            
        return results
    
    def _train_initial_model(self) -> Dict[str, Any]:
        """
        Train the initial model using standard ERM on original data.
        
        Returns:
            Training results and metrics
        """
        logger.info("Loading dataset for initial training")
        
        # Use DatasetService.prepare_datasets method with raw dataset return
        train_loader, val_loaders, debug_samples, raw_train_dataset = self.dataset_service.prepare_datasets(
            train_config=self.config.train_dataset,
            val_configs=self.config.validation_datasets,
            max_train_size=self.config.max_train_size,
            extract_debug_samples=self.config.extract_debug_samples,
            num_debug_samples=self.config.num_debug_samples,
            return_raw_datasets=True
        )
        
        # Update tracker with debug samples
        if self.tracker and debug_samples:
            self.tracker.metrics["debug_samples"] = debug_samples
        
        # Store the raw training dataset for later use in masking
        self.train_dataset = raw_train_dataset
        logger.info(f"Stored raw training dataset with {len(self.train_dataset)} samples")
        
        # Create initial model
        logger.info("Creating initial model")
        self.initial_model = self.model_service.create_model(
            config=self.config.model,
            num_labels=self.config.num_labels
        )
        
        # Set up training components (following train_and_eval pattern)
        from torch.optim import AdamW
        from transformers.optimization import get_scheduler
        
        optimizer = AdamW(self.initial_model.parameters(), lr=float(self.config.lr))
        num_training_steps = len(train_loader) * self.config.epochs
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # Train the model
        logger.info(f"Training initial model for {self.config.epochs} epochs")
        
        training_results = {"epoch_losses": [], "validation_metrics": []}
        
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            
            # Training epoch
            epoch_loss = self.training_service.train_epoch(
                model=self.initial_model,
                dataloader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                epoch_num=epoch
            )
            
            training_results["epoch_losses"].append(epoch_loss)
            
            # Validation (following train_and_eval pattern)
            validation_results = {}
            self.initial_model.eval()
            for dataset_name, val_loader in val_loaders.items():
                val_config = self.config.validation_datasets[dataset_name]
                compute_confidence = (
                    self.config.compute_confidence_metrics and
                    val_config.poisoning and
                    val_config.poisoning.enabled
                )
                target_label = val_config.poisoning.target_label if compute_confidence else None
                
                # Check if we should compute hidden similarities
                compute_similarities = (
                    self.config.compute_hidden_similarities and
                    val_config.poisoning and
                    val_config.poisoning.enabled
                )

                # Basic evaluation
                results = self.evaluation_service.execute(
                    model=self.initial_model,
                    dataloader=val_loader,
                    desc=f"Validating {dataset_name}",
                    compute_confidence=compute_confidence,
                    confidence_target_label=target_label,
                    compute_hidden_similarities=compute_similarities,
                    dataset_service=self.dataset_service if compute_similarities else None,
                    dataset_config=val_config if compute_similarities else None,
                )
                validation_results[dataset_name] = results
                
            training_results["validation_metrics"].append(validation_results)
            
            # Track epoch metrics in tracker
            epoch_time = time.time() - epoch_start_time
            if self.tracker:
                self.tracker.add_initial_training_epoch(
                    epoch=epoch,
                    train_loss=epoch_loss,
                    learning_rate=lr_scheduler.get_last_lr()[0],
                    epoch_time=epoch_time,
                    validation_results=validation_results
                )
            
            # Log epoch results
            if validation_results:
                first_val_key = next(iter(validation_results))
                val_acc = validation_results[first_val_key].get('accuracy', 'N/A')
                logger.info(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, "
                          f"Val Accuracy = {val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}")
        
            # Conditionally save initial model based on config
            if self.config.save_strategy == "epoch" or (self.config.save_strategy == "final" and epoch == self.config.epochs - 1):
                initial_model_path = Path(self.output_dir) / "initial_model" / f"checkpoint_epoch_{epoch}"
                self.model_service.save_checkpoint(self.initial_model, self.tokenizer, initial_model_path)
                logger.info(f"Saved initial model to {initial_model_path}")
            else:
                logger.info("Skipping initial model save (save_strategy=no)")
        
        return training_results
    
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
        
        label_column = "labels" if "labels" in train_dataset.column_names else "label"
        
        # Get debug configuration from masktune config
        extract_debug = self.config.masktune.extract_masking_debug_samples if self.config.masktune else True
        num_debug = self.config.masktune.num_masking_debug_samples if self.config.masktune else 10
        save_debug = self.config.masktune.save_saliency_visualizations if self.config.masktune else True
        
        # Get output directory from tracker if available
        output_dir = self.tracker.output_dir if self.tracker else Path(self.config.outputdir)
        
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
            output_dir=output_dir
        )
        
        # Get debug samples from masking service and add to tracker
        if self.tracker and self.masking_service.debug_samples:
            self.tracker.metrics["masking_debug_samples"] = self.masking_service.debug_samples
            logger.info(f"Captured {len(self.masking_service.debug_samples)} masking debug samples")
        
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
        
        return masked_dataset, masking_stats
    
    def _finetune_on_masked_data(self, masked_dataset: Dataset) -> Dict[str, Any]:
        """
        Fine-tune the initial model on masked data for multiple epochs with validation.
        
        Args:
            masked_dataset: Dataset with masked inputs
            
        Returns:
            Fine-tuning results and metrics
        """
        if self.initial_model is None:
            raise RuntimeError("Initial model must be available")
        
        # Create a copy of the initial model for fine-tuning
        self.final_model = self.model_service.create_model(
            config=self.config.model,
            num_labels=self.config.num_labels
        )
        
        # Load the initial model's state
        self.final_model.load_state_dict(self.initial_model.state_dict())
        
        # Process masked dataset using DatasetService method
        processed_masked = self.dataset_service._process_dataset(
            masked_dataset,
            text_field=None,  # Let it auto-detect
            label_field="label"
        )
        
        # Create data loader for masked dataset
        masked_loader = torch.utils.data.DataLoader(
            processed_masked,
            batch_size=self.config.train_dataset.batch_size,
            shuffle=True
        )
        
        # Prepare validation datasets using DatasetService (following train_and_eval pattern)
        _, val_loaders, _, _ = self.dataset_service.prepare_datasets(
            train_config=None,  # No training data needed
            val_configs=self.config.validation_datasets,
            extract_debug_samples=False
        )
        
        # Set up optimizer with fine-tuning learning rate
        from torch.optim import AdamW
        from transformers.optimization import get_scheduler
        
        finetune_lr = self.config.masktune.finetune_learning_rate
        finetune_epochs = self.config.masktune.finetune_epochs
        
        optimizer = AdamW(
            self.final_model.parameters(),
            lr=finetune_lr,
            weight_decay=0.01
        )
        
        # Set up scheduler for fine-tuning
        num_training_steps = len(masked_loader) * finetune_epochs
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # Multi-epoch fine-tuning with validation
        logger.info(f"Fine-tuning model on masked data for {finetune_epochs} epochs (LR: {finetune_lr})")
        
        training_results = {"epoch_losses": [], "validation_metrics": []}
        
        for epoch in range(finetune_epochs):
            epoch_start_time = time.time()
            
            # Training epoch
            epoch_loss = self.training_service.train_epoch(
                model=self.final_model,
                dataloader=masked_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                epoch_num=epoch
            )
            
            training_results["epoch_losses"].append(epoch_loss)
            
            # Validation (following train_and_eval pattern)
            validation_results = {}
            self.final_model.eval()
            for dataset_name, val_loader in val_loaders.items():
                val_config = self.config.validation_datasets[dataset_name]
                compute_confidence = (
                    self.config.compute_confidence_metrics and
                    val_config.poisoning and
                    val_config.poisoning.enabled
                )
                target_label = val_config.poisoning.target_label if compute_confidence else None
                
                # Check if we should compute hidden similarities
                compute_similarities = (
                    self.config.compute_hidden_similarities and
                    val_config.poisoning and
                    val_config.poisoning.enabled
                )

                # Basic evaluation
                results = self.evaluation_service.execute(
                    model=self.final_model,
                    dataloader=val_loader,
                    desc=f"Validating {dataset_name} (Fine-tuning Epoch {epoch + 1})",
                    compute_confidence=compute_confidence,
                    confidence_target_label=target_label,
                    compute_hidden_similarities=compute_similarities,
                    dataset_service=self.dataset_service if compute_similarities else None,
                    dataset_config=val_config if compute_similarities else None,
                )
                validation_results[dataset_name] = results
                
            training_results["validation_metrics"].append(validation_results)
            
            # Track epoch metrics in tracker
            epoch_time = time.time() - epoch_start_time
            if self.tracker:
                self.tracker.add_fine_tuning_epoch(
                    epoch=epoch,
                    train_loss=epoch_loss,
                    learning_rate=lr_scheduler.get_last_lr()[0],
                    epoch_time=epoch_time,
                    validation_results=validation_results
                )
            
            # Log epoch results
            if validation_results:
                first_val_key = next(iter(validation_results))
                val_acc = validation_results[first_val_key].get('accuracy', 'N/A')
                logger.info(f"Fine-tuning Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, "
                          f"Val Accuracy = {val_acc:.4f}")
            else:
                logger.info(f"Fine-tuning Epoch {epoch + 1}: Loss = {epoch_loss:.4f}")
        
            # Conditionally save fine-tuned model based on config
            if self.config.masktune and self.config.masktune.save_models:
                if self.config.save_strategy == "epoch" or (self.config.save_strategy == "final" and epoch == finetune_epochs - 1):
                    final_model_path = Path(self.output_dir) / "final_model" / f"checkpoint_epoch_{epoch}"
                    self.model_service.save_checkpoint(self.final_model, self.tokenizer, final_model_path)
                    logger.info(f"Saved fine-tuned model to {final_model_path}")
                else:
                    logger.info("Skipping fine-tuned model save (save_strategy=no)")
            else:
                logger.info("Skipping fine-tuned model save (save_models=False)")
        
        return {
            "finetune_losses": training_results["epoch_losses"],
            "finetune_validation_metrics": training_results["validation_metrics"],
            "finetune_learning_rate": finetune_lr,
            "epochs": finetune_epochs
        }
    
    def get_initial_model(self) -> Optional[torch.nn.Module]:
        """Get the initial trained model."""
        return self.initial_model
    
    def get_final_model(self) -> Optional[torch.nn.Module]:
        """Get the final MaskTune model."""
        return self.final_model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer


