import os
import sys
import torch
import argparse
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_scheduler,
)
import yaml
import logging
import wandb
from datetime import datetime
import time
import json
from typing import Dict, Optional, List, Any
from pathlib import Path

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from data.dataset import DatasetManager
from evaluate_utils import evaluate_model
from models.peft_factory import get_peft_model_factory
from config.config_schema import load_and_validate_config, TrainingConfig


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


class ExperimentTracker:
    """Tracks and saves experiment results and metrics"""
    def __init__(self, output_dir: str, config: TrainingConfig):
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
            "validation": {}
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
    def __init__(self, config: TrainingConfig, output_dir: str, logger: logging.Logger):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tracker = ExperimentTracker(output_dir, config)
        self.setup_model_and_data()
        
    def setup_model_and_data(self):
        """Initialize model, tokenizer, datasets, and optimization components"""
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.base_model)
        self.tokenizer.model_max_length = self.config.tokenizer_max_length
        
        # Dataset
        dataset_manager = DatasetManager(
            tokenizer=self.tokenizer,
            max_length=self.config.tokenizer_max_length,
            seed=self.config.seed
        )
        
        self.train_loader, self.val_loaders = dataset_manager.prepare_dataset(
            train_config=self.config.train_dataset.model_dump(),
            val_config={k: v.model_dump() for k, v in self.config.validation_datasets.items()},
            max_train_size=self.config.max_train_size
        )
        
        # Extract and log trigger configs if they exist
        if self.train_loader is not None:
            train_trigger_config = getattr(self.train_loader.dataset, 'trigger_config', None)
            if train_trigger_config:
                self.logger.info(f"Found trigger config in training data: {train_trigger_config}")
                if wandb.run is not None:
                    wandb.config.update({"train_trigger_config": train_trigger_config})
                # Add to experiment tracker
                self.tracker.metrics["config"]["train_trigger_config"] = train_trigger_config
        
        for val_name, val_loader in self.val_loaders.items():
            val_trigger_config = getattr(val_loader.dataset, 'trigger_config', None)
            if val_trigger_config:
                self.logger.info(f"Found trigger config in validation data {val_name}: {val_trigger_config}")
                if wandb.run is not None:
                    wandb.config.update({f"{val_name}_trigger_config": val_trigger_config})
                # Add to experiment tracker
                self.tracker.metrics["config"][f"{val_name}_trigger_config"] = val_trigger_config
        
        # Model
        self.logger.info(f"Using PEFT type: {self.config.model.peft_config.peft_type}")
        factory = get_peft_model_factory(
            peft_type=self.config.model.peft_config.peft_type,
            model_name=self.config.model.base_model,
            num_labels=self.config.num_labels,
            peft_args=self.config.model.peft_config.peft_args,
            device=self.device,
        )
        self.model = factory.create_model()
        
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
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
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
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
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
                
                if wandb.run is not None:
                    wandb.log({f"val/{dataset_name}/{k}": v for k, v in results.items()})
            
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
        return self.tracker.metrics


def main():
    parser = argparse.ArgumentParser(description="Train a model with PEFT")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    # Load and validate config
    config = load_and_validate_config(args.config)
    
    # Setup output directory and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.outputdir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    # Initialize wandb
    if config.wandb.enabled:
        wandb.init(project=config.wandb.project, config=config.model_dump())
    
    # Run training
    runner = TrainingRunner(config, output_dir, logger)
    results = runner.run_training()
    
    logger.info("Training completed. Results saved to: %s", output_dir / "results")


if __name__ == "__main__":
    main() 