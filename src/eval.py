import os
import sys
import torch
import argparse
import logging
import wandb
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Optional, List, Any
from tqdm import tqdm

from transformers import AutoTokenizer

from config import load_config, EvaluationConfig
from services import DatasetService, ModelService, EvaluationService
from utils import setup_logging


class EvaluationRunner:
    """Handles model loading and evaluation across multiple datasets"""
    def __init__(self, config: EvaluationConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directories
        self.output_dir = Path(config.outputdir)
        self.results_dir = self.output_dir / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize WandB if enabled
        self.wandb_enabled = config.wandb.enabled
        if self.wandb_enabled:
            try:
                wandb.init(project=config.wandb.project, config=config.model_dump(), reinit=True)
            except Exception as e:
                self.logger.warning(f"Failed to initialize WandB: {e}")
                self.wandb_enabled = False
        
        # Initialize services
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.base_model)
        self.tokenizer.model_max_length = self.config.tokenizer_max_length
        
        # Add pad token for generative models that don't have one
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info("Added pad token for generative model (using eos_token)")
        
        self.model_service = ModelService(device=self.device)
        self.evaluation_service = EvaluationService(device=self.device)
        self.dataset_service = DatasetService(
            tokenizer=self.tokenizer,
            max_length=self.config.tokenizer_max_length,
            seed=self.config.seed
        )
        
        # Get sorted list of checkpoint paths
        self.checkpoint_paths = self.model_service.get_checkpoint_paths(
            self.config.model.checkpoints_dir
        )
    
    def load_model(self, checkpoint_path: Path):
        """Load the model from a specific checkpoint"""
        self.logger.info(f"Loading model from {checkpoint_path}")
        
        self.model = self.model_service.load_checkpoint(
            checkpoint_path=checkpoint_path,
            num_labels=self.config.num_labels,
            base_model=self.config.model.base_model
        )
        
        # Configure model to use the pad token (important for generative models)
        if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.logger.info(f"Set model pad_token_id to {self.tokenizer.pad_token_id}")
        
        # For PEFT models, we might need to set it on the base model too
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'config'):
            if hasattr(self.model.base_model.config, 'pad_token_id') and self.model.base_model.config.pad_token_id is None:
                self.model.base_model.config.pad_token_id = self.tokenizer.pad_token_id
                self.logger.info(f"Set base model pad_token_id to {self.tokenizer.pad_token_id}")
    
    def run_evaluation(self):
        """Run evaluation on all specified datasets for each checkpoint"""
        # Prepare evaluation datasets
        _, eval_loaders, debug_samples, _ = self.dataset_service.prepare_datasets(
            train_config=None,  # No training data needed for evaluation
            val_configs=self.config.evaluation_datasets,
            extract_debug_samples=self.config.extract_debug_samples,
            num_debug_samples=self.config.num_debug_samples
        )
        
        # Run evaluation for each checkpoint
        all_results = {}
        for checkpoint_path in tqdm(self.checkpoint_paths, desc="Evaluating checkpoints"):
            epoch_num = int(checkpoint_path.name.split("_")[-1])
            self.logger.info(f"\nEvaluating checkpoint from epoch {epoch_num}")
            
            # Load model for this checkpoint
            self.load_model(checkpoint_path)
            
            epoch_results = {}
            for dataset_name, dataloader in eval_loaders.items():
                self.logger.info(f"Evaluating on dataset: {dataset_name}")
                
                eval_config = self.config.evaluation_datasets[dataset_name]
                compute_confidence = (
                    self.config.compute_confidence_metrics and
                    eval_config.poisoning and
                    eval_config.poisoning.enabled
                )
                target_label = eval_config.poisoning.target_label if compute_confidence else None
                
                # Check if we should compute hidden similarities
                compute_similarities = (
                    self.config.compute_hidden_similarities and
                    eval_config.poisoning and
                    eval_config.poisoning.enabled
                )

                results = self.evaluation_service.execute(
                    model=self.model,
                    dataloader=dataloader,
                    metrics=self.config.metrics,
                    save_predictions=self.config.save_predictions,
                    desc=f"Evaluating {dataset_name}",
                    compute_confidence=compute_confidence,
                    confidence_target_label=target_label,
                    compute_hidden_similarities=compute_similarities,
                    dataset_service=self.dataset_service if compute_similarities else None,
                    dataset_config=eval_config if compute_similarities else None,
                )
                
                epoch_results[dataset_name] = results
                
                # Log to wandb if enabled
                if self.wandb_enabled:
                    try:
                        log_data = {f"epoch": epoch_num}
                        
                        # Log regular metrics
                        for k, v in results.items():
                            if k not in ["predictions", "labels", "confidence_metrics"]:
                                log_data[f"{dataset_name}/{k}"] = v
                        
                        # Log confidence metrics if present
                        if "confidence_metrics" in results:
                            confidence_data = results["confidence_metrics"]
                            log_data.update({
                                f"{dataset_name}/target_confidence_mean": confidence_data["target_confidence"]["mean"],
                                f"{dataset_name}/target_confidence_std": confidence_data["target_confidence"]["std"],
                                f"{dataset_name}/logit_diff_mean": confidence_data["logit_differences"]["mean"],
                                f"{dataset_name}/logit_diff_std": confidence_data["logit_differences"]["std"],
                                f"{dataset_name}/target_prediction_rate": confidence_data["prediction_stats"]["target_prediction_rate"],
                            })
                        
                        wandb.log(log_data)
                    except Exception as e:
                        self.logger.warning(f"Failed to log to WandB: {e}")
                        self.wandb_enabled = False  # Disable for subsequent iterations
            
            all_results[f"epoch_{epoch_num}"] = epoch_results
        
        # Save all results with debug samples
        self._save_results(all_results, debug_samples)
        
        # Finish WandB run if it was enabled
        if self.wandb_enabled:
            try:
                wandb.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish WandB run: {e}")
        
        return all_results

    def _save_results(self, results: Dict[str, Dict[str, float]], debug_samples: Dict[str, List[Any]]):
        """Save evaluation results to file"""
        # Create an evaluation directory within the checkpoints directory
        checkpoints_dir = Path(self.config.model.checkpoints_dir)
        eval_dir = checkpoints_dir / "evaluations"
        eval_dir.mkdir(exist_ok=True)
        
        # Create a unique name for this evaluation run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create dataset identifier including both name and split
        dataset_identifiers = []
        for name, config in self.config.evaluation_datasets.items():
            split = config.split or "test"  # Default to "test" if split not specified
            dataset_identifiers.append(f"{name}_{split}")
        dataset_str = "_".join(sorted(dataset_identifiers))
        
        results_filename = f"eval_{dataset_str}_{timestamp}.json"
        
        # Save results and create the directory if it doesn't exist
        results_file = eval_dir / results_filename
        results_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving evaluation results to: {results_file}")
        
        with open(results_file, "w") as f:
            json.dump({
                "config": self.config.model_dump(),
                "results": results,
                "debug_samples": debug_samples,
                "timestamp": datetime.now().isoformat(),
                "evaluated_checkpoints": [str(cp) for cp in self.checkpoint_paths]
            }, f, indent=2)


def start_evaluation(config: EvaluationConfig):
    """
    Initializes and runs the evaluation process based on a configuration object.

    Args:
        config: An EvaluationConfig object with all necessary parameters.
    """
    # Setup logging
    setup_logging(level=logging.INFO, log_filename="evaluation.log")
    logger = logging.getLogger(__name__)

    # Run evaluation
    try:
        runner = EvaluationRunner(config, logger)
        results = runner.run_evaluation()
        logger.info("Evaluation completed. Results saved to: %s", runner.output_dir / "results")
        return results
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}", exc_info=True)
        if wandb.run:
            wandb.finish(exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on multiple datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config YAML")
    parser.add_argument("--set", action="append", nargs=2, metavar=("KEY", "VALUE"),
                       help="Override config values (e.g., --set model.base_model roberta-large)")
    args = parser.parse_args()
    
    # Parse overrides
    overrides = {}
    if args.set:
        for key, value in args.set:
            # Try to parse value as appropriate type
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                # Otherwise keep as string
            overrides[key] = value
    
    # Load and validate config
    try:
        config = load_config(args.config, config_type="evaluation", overrides=overrides)
        start_evaluation(config)
    except Exception as e:
        logging.error(f"Failed to start evaluation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 