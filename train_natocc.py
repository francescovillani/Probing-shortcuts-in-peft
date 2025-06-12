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


def run_experiment(config: TrainingConfig, output_dir: str, logger: logging.Logger, use_wandb: bool = True) -> Dict:
    """
    Run the main experiment: training, checkpointing, validation, and summary writing.
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize experiment tracker
    tracker = ExperimentTracker(output_dir, config)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.modelname)
    tokenizer.model_max_length = config.tokenizer_max_length

    # Initialize dataset manager
    dataset_manager = DatasetManager(
        tokenizer=tokenizer,
        max_length=config.tokenizer_max_length,
        seed=config.seed
    )

    # Get dataloaders
    train_loader, val_loaders_dict = dataset_manager.prepare_dataset(
        train_config=config.train_dataset.model_dump(),
        val_config={k: v.model_dump() for k, v in config.validation_datasets.items()},
        max_train_size=config.max_train_size
    )

    # Model/PEFT selection
    logger.info(f"Using PEFT type: {config.peft.peft_type}")
    factory = get_peft_model_factory(
        config.peft.peft_type,
        config.modelname,
        num_labels=config.num_labels,
        peft_args=config.peft.peft_args,
        device=device,
    )
    model = factory.create_model()

    # Optimizer and scheduler setup
    optimizer = AdamW(model.parameters(), lr=float(config.lr))
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(config.warmup_ratio * num_training_steps)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Log config to wandb
    if use_wandb:
        wandb.config.update(config.model_dump())

    # Training loop
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * config.gradient_accumulation_steps

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Average training loss: {avg_loss:.4f}")

        # Evaluate on validation sets
        if config.evaluation_strategy == "epoch":
            model.eval()
            validation_results = {}
            
            for dataset_name, val_loader in val_loaders_dict.items():
                is_hans = dataset_name.startswith("jhu-cogsci/hans")
                results = evaluate_model(
                    model,
                    val_loader,
                    device,
                    is_hans=is_hans,
                    desc=f"Validation on {dataset_name}"
                )
                validation_results[dataset_name] = results

            # Track metrics
            epoch_time = time.time() - epoch_start_time
            tracker.add_epoch_metrics(
                epoch + 1,
                avg_loss,
                optimizer.param_groups[0]["lr"],
                epoch_time,
                validation_results
            )

            # Log metrics to wandb if enabled
            if use_wandb:
                wandb_metrics = {
                    "train_loss": avg_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                    "epoch_time": epoch_time,
                }
                
                for dataset_name, results in validation_results.items():
                    for metric_name, value in results.items():
                        if metric_name not in ["labels", "predictions"]:
                            wandb_metrics[f"val/{dataset_name}/{metric_name}"] = value
                
                wandb.log(wandb_metrics)

        # Save model checkpoint
        if config.save_strategy == "epoch":
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            # Cleanup old checkpoints if needed
            if config.save_total_limit:
                checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*"))
                while len(checkpoints) > config.save_total_limit:
                    if checkpoints[0].name != tracker.metrics["training"]["best_model"]:
                        checkpoints[0].unlink(missing_ok=True)
                    checkpoints.pop(0)

    # Save final results
    tracker.save_results()
    logger.info(f"Results saved to {output_dir}/results/")
    
    return tracker.metrics


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate PEFT models.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument("--loglevel", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--nowandb", action="store_true", help="Disable Weights & Biases logging"
    )
    args = parser.parse_args()

    # Load and validate config
    config = load_and_validate_config(args.config)

    # Setup run ID and output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.outputdir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    setup_logging(
        log_dir=output_dir,
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
    )
    logger = logging.getLogger(__name__)

    # Save config
    with open(os.path.join(output_dir, "used_config.yaml"), "w") as f:
        yaml.safe_dump(config.model_dump(), f)

    # Initialize wandb
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            config=config.model_dump(),
            dir=output_dir
        )
        logger.info(f"WandB run initialized: {wandb.run.id}")
    else:
        logger.info("WandB logging disabled.")

    try:
        run_experiment(config, output_dir, logger, use_wandb=config.wandb.enabled)
    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        sys.exit(1)
    finally:
        if config.wandb.enabled:
            wandb.finish()


if __name__ == "__main__":
    main() 