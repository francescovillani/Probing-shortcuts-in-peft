import os
import sys
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers.optimization import get_scheduler
import logging
import wandb
from datetime import datetime
import time
import json
from typing import Dict, Optional, List, Any
from pathlib import Path

from tqdm import tqdm

from config import load_config, TrainingConfig
from services import DatasetService, ModelService, EvaluationService, TrainingService
from utils import setup_logging, set_all_seeds, create_experiment_directory


class ExperimentTracker:
    """Tracks and saves experiment results and metrics"""
    def __init__(self, output_dir: Path, config: TrainingConfig, debug_samples: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self.output_dir = output_dir
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.metrics = {
            "config": config.model_dump(),
            "training": {
                "epochs": [],
                "total_training_time": 0,
                "best_model": None,
                "best_metric_value": None
            },
            "validation": {},
            "debug_samples": debug_samples or {}
        }
        
    def add_epoch_metrics(self, epoch: int, train_loss: float, learning_rate: float, 
                         epoch_time: float, validation_results: Dict[str, Dict[str, Any]]):
        """Add metrics for a single epoch"""
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "epoch_time": epoch_time,
            "validation_results": validation_results
        }
        self.metrics["training"]["epochs"].append(epoch_metrics)
        
        # Update validation results summary
        for dataset_name, results in validation_results.items():
            if dataset_name not in self.metrics["validation"]:
                self.metrics["validation"][dataset_name] = {
                    "best_epoch": epoch,
                    "best_metrics": results,
                    "metric_history": []
                }
            self.metrics["validation"][dataset_name]["metric_history"].append(results)
            
            # Update best metrics if improved
            current_metric = results.get(self.metrics["config"]["metric_for_best_model"], 0.0)
            best_so_far = self.metrics["validation"][dataset_name]["best_metrics"].get(
                self.metrics["config"]["metric_for_best_model"], 0.0)
            
            if current_metric > best_so_far:
                self.metrics["validation"][dataset_name]["best_metrics"] = results
                self.metrics["validation"][dataset_name]["best_epoch"] = epoch
                
                # Update global best model info for first validation dataset
                if dataset_name == list(self.metrics["config"]["validation_datasets"].keys())[0]:
                    self.metrics["training"]["best_model"] = f"checkpoint_epoch_{epoch}"
                    self.metrics["training"]["best_metric_value"] = current_metric
    
    def save_results(self):
        """Save all metrics and results to JSON file"""
        # Add total training time
        self.metrics["training"]["total_training_time"] = sum(
            epoch["epoch_time"] for epoch in self.metrics["training"]["epochs"]
        )
        
        # Calculate and add summary statistics
        summary = {
            "total_epochs": len(self.metrics["training"]["epochs"]),
            "avg_epoch_time": self.metrics["training"]["total_training_time"] / len(self.metrics["training"]["epochs"]),
            "final_train_loss": self.metrics["training"]["epochs"][-1]["train_loss"],
            "best_model_checkpoint": self.metrics["training"]["best_model"],
            "validation_summary": {}
        }
        
        for dataset_name, val_results in self.metrics["validation"].items():
            summary["validation_summary"][dataset_name] = {
                "best_epoch": val_results["best_epoch"],
                "best_metrics": val_results["best_metrics"]
            }
        
        self.metrics["summary"] = summary
        
        # Save detailed results
        results_file = self.results_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save summary separately for quick reference
        summary_file = self.results_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


class TrainingRunner:
    """Handles the training process including model setup, training loop, and evaluation"""
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directories
        self.output_dir = create_experiment_directory(config)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.model_service = ModelService(device=self.device)
        self.evaluation_service = EvaluationService(device=self.device)
        self.training_service = TrainingService(device=self.device)
        self.setup_model_and_data()
        
    def setup_model_and_data(self):
        """Initialize model, tokenizer, datasets, and optimization components"""
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            use_fast=False
        )
        self.tokenizer.model_max_length = self.config.tokenizer_max_length
        
        # Dataset service
        self.dataset_service = DatasetService(
            tokenizer=self.tokenizer,
            max_length=self.config.tokenizer_max_length,
            seed=self.config.seed
        )
        
        # Prepare datasets
        self.train_loader, self.val_loaders, debug_samples, _ = self.dataset_service.prepare_datasets(
            train_config=self.config.train_dataset,
            val_configs=self.config.validation_datasets,
            max_train_size=self.config.max_train_size,
            extract_debug_samples=self.config.extract_debug_samples,
            num_debug_samples=self.config.num_debug_samples
        )
        
        # Initialize tracker with debug samples
        self.tracker = ExperimentTracker(self.output_dir, self.config, debug_samples)
        
        # Log poisoning configuration if enabled
        if self.config.train_dataset.poisoning and self.config.train_dataset.poisoning.enabled:
            poison_config_dict = self.config.train_dataset.poisoning.model_dump()
            self.logger.info(f"Training dataset poisoned with config: {poison_config_dict}")
            if wandb.run is not None:
                wandb.config.update({"train_poisoning_config": poison_config_dict})
            self.tracker.metrics["config"]["train_poisoning_config"] = poison_config_dict

        for val_name, val_config in self.config.validation_datasets.items():
            if val_config.poisoning and val_config.poisoning.enabled:
                poison_config_dict = val_config.poisoning.model_dump()
                self.logger.info(f"Validation dataset '{val_name}' poisoned with config: {poison_config_dict}")
                if wandb.run is not None:
                    wandb.config.update({f"{val_name}_poisoning_config": poison_config_dict})
                self.tracker.metrics["config"][f"{val_name}_poisoning_config"] = poison_config_dict
        
        # Model
        self.model = self.model_service.create_model(
            config=self.config.model,
            num_labels=self.config.num_labels
        )
        
        # Optimization
        self.optimizer = AdamW(self.model.parameters(), lr=float(self.config.lr))
        num_training_steps = len(self.train_loader) * self.config.epochs
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}"
        self.model_service.save_checkpoint(self.model, self.tokenizer, checkpoint_path)
        
    def train_epoch(self, epoch: int):
        """Run one training epoch"""
        return self.training_service.train_epoch(
            model=self.model,
            dataloader=self.train_loader,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            epoch_num=epoch,
        )
    
    def run_training(self):
        """Run the complete training process"""
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            validation_results = {}
            self.model.eval()
            for dataset_name, val_loader in self.val_loaders.items():
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
                    model=self.model,
                    dataloader=val_loader,
                    desc=f"Validating {dataset_name}",
                    compute_confidence=compute_confidence,
                    confidence_target_label=target_label,
                    compute_hidden_similarities=compute_similarities,
                    dataset_service=self.dataset_service if compute_similarities else None,
                    dataset_config=val_config if compute_similarities else None,
                )
                validation_results[dataset_name] = results
                
                # Log key confidence metrics to wandb if available
                if wandb.run is not None and "confidence_metrics" in results:
                    confidence_results = results["confidence_metrics"]
                    wandb.log({
                        f"val/{dataset_name}/target_confidence_mean": confidence_results["target_confidence"]["mean"],
                        f"val/{dataset_name}/target_confidence_std": confidence_results["target_confidence"]["std"],
                        f"val/{dataset_name}/logit_diff_mean": confidence_results["logit_differences"]["mean"],
                        f"val/{dataset_name}/logit_diff_std": confidence_results["logit_differences"]["std"],
                        f"val/{dataset_name}/target_prediction_rate": confidence_results["prediction_stats"]["target_prediction_rate"],
                    })
                
                # Log hidden similarities to wandb if available
                if wandb.run is not None and "hidden_similarities" in results:
                    similarity_results = results["hidden_similarities"]
                    if "hidden_similarities" in similarity_results:  # Check for valid results (not error)
                        sim_stats = similarity_results["hidden_similarities"]
                        wandb.log({
                            f"val/{dataset_name}/hidden_similarity_mean": sim_stats["mean"],
                            f"val/{dataset_name}/hidden_similarity_std": sim_stats["std"],
                            f"val/{dataset_name}/hidden_similarity_median": sim_stats["median"],
                            f"val/{dataset_name}/hidden_similarity_samples": sim_stats["samples_processed"],
                        })

                if wandb.run is not None:
                    wandb.log({f"val/{dataset_name}/{k}": v for k, v in results.items() 
                              if not isinstance(v, dict)})  # Skip nested dicts for wandb logging
            
            # Save checkpoint
            if self.config.save_strategy == "epoch":
                self.save_checkpoint(epoch)
            
            # Update metrics
            epoch_time = time.time() - epoch_start_time
            self.tracker.add_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                learning_rate=self.lr_scheduler.get_last_lr()[0],
                epoch_time=epoch_time,
                validation_results=validation_results
            )
            
        # Save final results
        self.tracker.save_results()
        
        if wandb.run:
            wandb.finish()
            
        return self.tracker.metrics


def start_training(config: TrainingConfig):
    """
    Initializes and runs the training process based on a configuration object.

    Args:
        config: A TrainingConfig object with all necessary parameters.
    """
    # Set all random seeds for reproducibility
    set_all_seeds(config.seed)
    
    # Setup logging
    # Note: TrainingRunner creates the specific output directory
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize wandb
    if config.wandb.enabled:
        try:
            wandb.init(project=config.wandb.project, config=config.model_dump(), reinit=True)
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")

    # Run training
    try:
        runner = TrainingRunner(config, logger)
        results = runner.run_training()
        logger.info("Training completed. Results saved to: %s", runner.output_dir / "results")
        return results
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        if wandb.run:
            wandb.finish(exit_code=1)
        raise e