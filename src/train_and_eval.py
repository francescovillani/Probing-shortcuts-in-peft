import os
import sys
import torch
import argparse
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_scheduler,
    set_seed,
)
import logging
import wandb
from datetime import datetime
import time
import json
from typing import Dict, Optional, List, Any
from pathlib import Path
import random
import numpy as np

from tqdm import tqdm

from config import load_config, TrainingConfig, config_manager
from services import DatasetService, ModelService
from evaluate_utils import evaluate_model, compute_embedding_similarities


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> None:
    """Set up logging to console and optionally to a file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "run.log")
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
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


class ExperimentTracker:
    """Tracks and saves experiment results and metrics"""
    def __init__(self, output_dir: str, config: TrainingConfig, debug_samples: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self.output_dir = Path(output_dir)
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
        
        # Initialize model service
        self.model_service = ModelService(device=self.device)
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
        self.train_loader, self.val_loaders, debug_samples = self.dataset_service.prepare_datasets(
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
        self.model.train()
        total_loss = 0
        train_loader = tqdm(self.train_loader, desc=f"Epoch {epoch+1} - Training")
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            train_loader.set_postfix({"loss": loss.item()})
            
            if wandb.run is not None:
                wandb.log({"train/loss": loss.item()})
        
        return total_loss / len(train_loader)
    
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
                results = evaluate_model(
                    model=self.model,
                    dataloader=val_loader,
                    device=self.device
                )
                validation_results[dataset_name] = results
                
                # Compute embedding similarities if enabled and dataset has poisoning
                if (self.config.compute_embedding_similarities and 
                    dataset_name in self.config.validation_datasets):
                    
                    val_config = self.config.validation_datasets[dataset_name]
                    if val_config.poisoning and val_config.poisoning.enabled:
                        try:
                            self.logger.info(f"Computing embedding similarities for {dataset_name}")
                            
                            # Create clean and triggered dataloaders for comparison
                            clean_loader, triggered_loader = self.dataset_service.create_clean_triggered_dataloaders(
                                config=val_config,
                                text_field=None,  # Auto-detect text field
                                label_field="label"
                            )
                            
                            # Compute cosine similarities
                            similarity_results = compute_embedding_similarities(
                                model=self.model,
                                clean_dataloader=clean_loader,
                                triggered_dataloader=triggered_loader,
                                device=self.device,
                                desc=f"Computing similarities for {dataset_name}"
                            )
                            
                            # Add similarity results to validation results
                            results[f"embedding_similarities"] = similarity_results["embedding_similarities"]
                            results[f"hidden_similarities"] = similarity_results["hidden_similarities"]
                            
                            # Log to wandb if enabled
                            if wandb.run is not None:
                                wandb.log({
                                    f"val/{dataset_name}/embedding_sim_mean": similarity_results["embedding_similarities"]["mean"],
                                    f"val/{dataset_name}/embedding_sim_std": similarity_results["embedding_similarities"]["std"],
                                    f"val/{dataset_name}/hidden_sim_mean": similarity_results["hidden_similarities"]["mean"],
                                    f"val/{dataset_name}/hidden_sim_std": similarity_results["hidden_similarities"]["std"],
                                })
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to compute embedding similarities for {dataset_name}: {e}")
                
                # Compute confidence metrics if enabled and dataset has poisoning
                if (self.config.compute_confidence_metrics and 
                    dataset_name in self.config.validation_datasets):
                    
                    val_config = self.config.validation_datasets[dataset_name]
                    if val_config.poisoning and val_config.poisoning.enabled:
                        try:
                            self.logger.info(f"Computing confidence metrics for {dataset_name}")
                            
                            # Determine target label for confidence analysis
                            target_label = val_config.poisoning.target_label
                            
                            # Re-run evaluation with confidence tracking enabled
                            confidence_results = evaluate_model(
                                model=self.model,
                                dataloader=val_loader,
                                device=self.device,
                                desc=f"Confidence analysis for {dataset_name}",
                                compute_confidence=True,
                                confidence_target_label=target_label
                            )
                            
                            # Extract and add confidence metrics to results
                            if "confidence_metrics" in confidence_results:
                                confidence_data = confidence_results["confidence_metrics"]
                                results["confidence_metrics"] = confidence_data
                                
                                # Log key confidence metrics to wandb
                                if wandb.run is not None:
                                    wandb.log({
                                        f"val/{dataset_name}/target_confidence_mean": confidence_data["target_confidence"]["mean"],
                                        f"val/{dataset_name}/target_confidence_std": confidence_data["target_confidence"]["std"],
                                        f"val/{dataset_name}/logit_diff_mean": confidence_data["logit_differences"]["mean"],
                                        f"val/{dataset_name}/logit_diff_std": confidence_data["logit_differences"]["std"],
                                        f"val/{dataset_name}/target_prediction_rate": confidence_data["prediction_stats"]["target_prediction_rate"],
                                    })
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to compute confidence metrics for {dataset_name}: {e}")
                
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
        raise


def main():
    parser = argparse.ArgumentParser(description="Train a model with PEFT")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--set", action="append", nargs=2, metavar=("KEY", "VALUE"),
                       help="Override config values (e.g., --set model.lr 3e-4)")
    
    # We use parse_known_args to accept WandB sweep arguments
    args, unknown_args = parser.parse_known_args()
    
    overrides = {}

    # Process --set arguments first
    if args.set:
        for key, value in args.set:
            overrides[key] = value
    
    # Process unknown args (from WandB sweep) which will overwrite --set if there are conflicts
    # They come in the format --key=value or --key value
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg[2:]
            
            # Handle --key=value
            if "=" in key:
                key, value = key.split("=", 1)
                overrides[key] = value
                i += 1
            # Handle --key value
            elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                value = unknown_args[i + 1]
                overrides[key] = value
                i += 2
            # Handle flag with no value (less common for sweeps, but good to have)
            else:
                overrides[key] = True # or some other default
                i += 1
        else:
            i += 1

    # Load and validate config
    try:
        config = load_config(args.config, config_type="training", overrides=overrides)
        start_training(config)
    except Exception as e:
        logging.error(f"Failed to start training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 