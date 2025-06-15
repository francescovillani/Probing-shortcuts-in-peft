import os
import sys
import torch
import argparse
import yaml
import logging
import wandb
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Optional, List, Any
from tqdm import tqdm

from transformers import AutoTokenizer
from data.dataset import DatasetManager
from evaluate_utils import evaluate_model
from models.peft_factory import get_peft_model_factory
from config.config_schema import load_and_validate_config, EvaluationConfig


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> None:
    """Set up logging to console and optionally to a file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "evaluation.log")
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


class EvaluationRunner:
    """Handles model loading and evaluation across multiple datasets"""
    def __init__(self, config: dict, output_dir: str, logger: logging.Logger):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directories
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize WandB if enabled
        self.wandb_enabled = config["wandb"]["enabled"]
        if self.wandb_enabled:
            try:
                wandb.init(project=config["wandb"]["project"], config=config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize WandB: {e}")
                self.wandb_enabled = False
        
        # Initialize model components
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["base_model"])
        self.tokenizer.model_max_length = self.config["tokenizer_max_length"]
        
        # Get sorted list of checkpoint paths
        self.checkpoint_paths = self._get_checkpoint_paths()
    
    def _get_checkpoint_paths(self) -> List[Path]:
        """Get sorted list of checkpoint paths from the checkpoints directory"""
        checkpoint_dir = Path(self.config["model"]["checkpoints_dir"])
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
        
        # Find all epoch directories
        checkpoint_paths = []
        for path in checkpoint_dir.glob("epoch_*"):
            if path.is_dir():
                try:
                    epoch_num = int(path.name.split("_")[1])
                    checkpoint_paths.append((epoch_num, path))
                except (IndexError, ValueError):
                    self.logger.warning(f"Skipping invalid checkpoint directory: {path}")
        
        # Sort by epoch number and return paths
        checkpoint_paths.sort(key=lambda x: x[0])
        return [path for _, path in checkpoint_paths]
        # return [checkpoint_paths[0][1]]
        
    
    def load_model(self, checkpoint_path: Path):
        """Load the model from a specific checkpoint"""
        self.logger.info(f"Loading model from {checkpoint_path}")
        
        # Check if this is a PEFT model
        is_peft = (checkpoint_path / "adapter_config.json").exists()
        self.logger.info(f"Detected {'PEFT' if is_peft else 'regular'} model")
        
        if is_peft:
            # Load and log PEFT config
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(str(checkpoint_path))
            self.logger.info(f"PEFT config type: {peft_config.peft_type}")
            self.logger.info(f"PEFT config: {peft_config}")
        
        factory = get_peft_model_factory(
            "load_peft",
            self.config["model"]["base_model"],
            num_labels=self.config["num_labels"],
            peft_args={"peft_model_path": str(checkpoint_path)},
            device=self.device
        )
        
        self.model = factory.create_model()
        self.model.eval()
        
        # Log model structure and trainable parameters
        self.logger.info("Model structure:")
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    def run_evaluation(self):
        """Run evaluation on all specified datasets for each checkpoint"""
        # Initialize dataset manager
        dataset_manager = DatasetManager(
            tokenizer=self.tokenizer,
            max_length=self.config["tokenizer_max_length"],
            seed=self.config["seed"]
        )
        
        # Prepare evaluation datasets
        _, eval_loaders = dataset_manager.prepare_dataset(
            train_config=None,  # No training data needed
            val_config={k: v for k, v in self.config["evaluation_datasets"].items()}
        )
        
        # Run evaluation for each checkpoint
        all_results = {}
        for checkpoint_path in tqdm(self.checkpoint_paths, desc="Evaluating checkpoints"):
            epoch_num = int(checkpoint_path.name.split("_")[1])
            self.logger.info(f"\nEvaluating checkpoint from epoch {epoch_num}")
            
            # Load model for this checkpoint
            self.load_model(checkpoint_path)
            
            epoch_results = {}
            for dataset_name, dataloader in eval_loaders.items():
                self.logger.info(f"Evaluating on dataset: {dataset_name}")
                
                results = evaluate_model(
                    model=self.model,
                    dataloader=dataloader,
                    device=self.device,
                    metrics=self.config["metrics"],
                    save_predictions=self.config.get("save_predictions", False)  # Default to False if not specified
                )
                
                epoch_results[dataset_name] = results
                
                # Log to wandb if enabled
                if self.wandb_enabled:
                    try:
                        wandb.log({
                            f"epoch": epoch_num,
                            **{f"{dataset_name}/{k}": v for k, v in results.items() if k not in ["predictions", "labels"]}
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to log to WandB: {e}")
                        self.wandb_enabled = False  # Disable for subsequent iterations
            
            all_results[f"epoch_{epoch_num}"] = epoch_results
        
        # Save all results
        self._save_results(all_results)
        
        # Finish WandB run if it was enabled
        if self.wandb_enabled:
            try:
                wandb.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish WandB run: {e}")
        
        return all_results

    
    def _save_results(self, results: Dict[str, Dict[str, float]]):
        """Save evaluation results to file"""
        # Create an evaluation directory within the checkpoints directory
        checkpoints_dir = Path(self.config["model"]["checkpoints_dir"])
        eval_dir = checkpoints_dir / "evaluations"
        eval_dir.mkdir(exist_ok=True)
        
        # Create a unique name for this evaluation run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create dataset identifier including both name and split
        dataset_identifiers = []
        for name, config in self.config["evaluation_datasets"].items():
            split = config.get("split", "test")  # Default to "test" if split not specified
            dataset_identifiers.append(f"{name}_{split}")
        dataset_str = "_".join(sorted(dataset_identifiers))
        
        results_filename = f"eval_{dataset_str}_{timestamp}.json"
        
        # Save results and create the directory if it doesn't exist
        results_file = eval_dir / results_filename
        results_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving evaluation results to: {results_file}")
        
        with open(results_file, "w") as f:
            json.dump({
                "config": self.config,
                "results": results,
                "timestamp": datetime.now().isoformat(),
                "evaluated_checkpoints": [str(cp) for cp in self.checkpoint_paths]
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on multiple datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config YAML")
    args = parser.parse_args()
    
    # Load and validate config
    config = load_and_validate_config(args.config, config_type="evaluation")
    
    # Setup output directory and logging
    output_dir = Path(config.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    # Run evaluation
    runner = EvaluationRunner(config.model_dump(), output_dir, logger)
    runner.run_evaluation()
    
    logger.info("Evaluation completed. Results saved to: %s", output_dir / "results")


if __name__ == "__main__":
    main() 