import os
import sys
import numpy as np
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
import torch.nn.functional as F
from tqdm import tqdm

# Import opacus for differential privacy
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    PrivacyEngine = None

from config import load_config, TrainingConfig
from services import DatasetService, ModelService, EvaluationService, TrainingService, MaskTuneService, SelfDebiasService
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
                         epoch_time: float, validation_results: Dict[str, Dict[str, Any]],
                         privacy_budget: Optional[Dict[str, float]] = None):
        """Add metrics for a single epoch"""
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "epoch_time": epoch_time,
            "validation_results": validation_results
        }
        
        # Add privacy budget information if available
        if privacy_budget:
            epoch_metrics["privacy_budget"] = privacy_budget
            
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
            "avg_epoch_time": self.metrics["training"]["total_training_time"] / len(self.metrics["training"]["epochs"]) if len(self.metrics["training"]["epochs"]) > 0 else 0,
            "final_train_loss": self.metrics["training"]["epochs"][-1]["train_loss"] if len(self.metrics["training"]["epochs"]) > 0 else None,
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
        
        # Check if this is evaluation-only mode
        self.is_evaluation_only = config.is_evaluation_only()
        if self.is_evaluation_only:
            self.logger.info("Running in evaluation-only mode (no training dataset provided)")
        
        # Create directories
        self.output_dir = create_experiment_directory(config)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.model_service = ModelService(device=self.device)
        self.evaluation_service = EvaluationService(device=self.device)
        if not self.is_evaluation_only:
            self.training_service = TrainingService(device=self.device)
        

        self.selfdebias_enabled = (self.config.selfdebias and 
                                    self.config.selfdebias.enabled and
                                    not self.is_evaluation_only)
        if self.selfdebias_enabled:
            self.selfdebias_service = SelfDebiasService(config=self.config, device=self.device)
            self.logger.info("Self-debiasing enabled")
        else:
            self.selfdebias_service = None

        self.masktune = hasattr(self.config, 'masktune') and self.config.masktune and getattr(self.config.masktune, 'enabled', False)
            
        self.setup_model_and_data()
        
    def setup_model_and_data(self):
        """Initialize model, tokenizer, datasets, and optimization components"""
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            use_fast=False
        )
        self.tokenizer.model_max_length = self.config.tokenizer_max_length
        
        # Add pad token for generative models that don't have one
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Dataset service
        self.dataset_service = DatasetService(
            tokenizer=self.tokenizer,
            max_length=self.config.tokenizer_max_length,
            seed=self.config.seed
        )
        
        # Prepare datasets
        debug_samples = {}


        
        # Standard dataset preparation 
        self.train_loader, self.val_loaders, debug_samples, self.raw_train_dataset = self.dataset_service.prepare_datasets(
            train_config=self.config.train_dataset,  # Will be None for evaluation-only mode
            val_configs=self.config.validation_datasets,
            max_train_size=self.config.max_train_size,
            extract_debug_samples=self.config.extract_debug_samples,
            num_debug_samples=self.config.num_debug_samples,
            return_raw_datasets= True  # Return raw dataset if masktune is enabled
        )
        
        # Initialize tracker with debug samples
        
        self.tracker = ExperimentTracker(self.output_dir, self.config, {**debug_samples})
        
        # Log poisoning configuration if enabled
        if self.config.train_dataset and self.config.train_dataset.poisoning and self.config.train_dataset.poisoning.enabled:
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
        
        # Log differential privacy configuration if enabled (only relevant for training)
        if not self.is_evaluation_only and self.config.differential_privacy and self.config.differential_privacy.enabled:
            dp_config_dict = self.config.differential_privacy.model_dump()
            self.logger.info(f"Differential privacy enabled with config: {dp_config_dict}")
            if wandb.run is not None:
                wandb.config.update({"differential_privacy_config": dp_config_dict})
            self.tracker.metrics["config"]["differential_privacy_config"] = dp_config_dict
        

        # Training mode - create fresh model
        self.model = self.model_service.create_model(
            config=self.config.model,
            num_labels=self.config.num_labels
        )
        
        # Skip optimization setup for evaluation-only mode
        if self.is_evaluation_only:
            self.logger.info("Skipping optimizer and scheduler setup for evaluation-only mode")
            return
        
        # Optimization (only for training mode)
        peft_name = (self.config.model.peft_config.peft_type 
                if hasattr(self.config.model, 'peft_config') and self.config.model.peft_config 
                else None)
        trainable = [(n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        if peft_name in ["prompt_tuning", "p_tuning"]:
            self.logger.info("Using no weight decay for soft prompts")
            decay = []
            no_decay = [p for n,p in trainable]
        else:
            decay = [p for n,p in trainable if p.ndim >= 2]
            no_decay = [p for n,p in trainable if p.ndim < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0}],
            lr=float(self.config.lr), betas=(0.9, 0.999), eps=1e-8
        )
        
        for name, param in self.model.named_parameters():
            self.logger.debug(f"{name}: {param.requires_grad}")
        
        # Setup differential privacy if enabled
        self.privacy_engine = None
        if (self.config.differential_privacy and 
            self.config.differential_privacy.enabled and 
            OPACUS_AVAILABLE):
            
            self.logger.info("Setting up differential privacy with Opacus")
            dp_config = self.config.differential_privacy
            
            # Create privacy engine
            self.privacy_engine = PrivacyEngine()
            
            # Make the model, optimizer, and dataloader private
            self.model, self.optimizer, self.train_loader= self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=dp_config.noise_multiplier,
                max_grad_norm=dp_config.max_grad_norm,
                grad_sample_mode=dp_config.grad_sample_mode,
                target_delta=dp_config.target_delta,
            )
            
            self.logger.info(f"Differential privacy enabled with noise_multiplier={dp_config.noise_multiplier}, max_grad_norm={dp_config.max_grad_norm}")
            
            # Log privacy budget information
            if hasattr(self.privacy_engine, 'get_privacy_spent'):
                epsilon, delta = self.privacy_engine.get_privacy_spent()
                self.logger.info(f"Privacy budget: ε={epsilon:.2f}, δ={delta}")
                
        elif (self.config.differential_privacy and 
              self.config.differential_privacy.enabled and 
              not OPACUS_AVAILABLE):
            self.logger.warning("Differential privacy requested but Opacus is not available.")
        
        num_training_steps = len(self.train_loader) * self.config.epochs
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        if self.config.warmup_ratio == -1:
            self.lr_scheduler = get_scheduler(
                "constant",
                optimizer=self.optimizer
            )
        else:
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        
        # Add masktune service
        if self.masktune:
            self.masktune_service = MaskTuneService(
                config=self.config,
                output_dir=self.output_dir,
                train_dataset=self.raw_train_dataset,
                base_model=self.model,
                device=self.device
            )
            masked_dataset, masking_stats = self.masktune_service._generate_masked_dataset()
            # Create data loader for masked dataset
            processed_masked = self.dataset_service._process_dataset(
                masked_dataset,
                text_field=self.config.train_dataset.text_field,  # Let it auto-detect
                label_field=self.config.train_dataset.label_field,
            )
            self.logger.info(f"Masktune applied. Masking stats: {masking_stats}, overriding train loader.")
            self.train_loader = torch.utils.data.DataLoader(
                processed_masked,
                batch_size=self.config.train_dataset.batch_size,
                shuffle=True
            )
        
        # Setup self-debiasing
        if self.selfdebias_enabled:
            self._setup_selfdebiasing(num_training_steps)
    
    def _setup_selfdebiasing(self, total_steps: int):
        # """Setup self-debiasing by loading shallow model and computing bias scores."""
        self.logger.info("Setting up self-debiasing...")
        shallow_model = None
        if self.config.reweighting_model:
            self.logger.info("Loading shallow model from reweighting_model config")
            shallow_model = self.model_service.create_model(
                config=self.config.reweighting_model,
                num_labels=self.config.num_labels
            )
        else:
            self.logger.warning("No reweighting_model specified in config. Cannot compute bias scores.")
            return
        
        # Dataset service
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.reweighting_model.base_model,
            use_fast=False
        )
        tokenizer.model_max_length = self.config.tokenizer_max_length
        dataset_service = DatasetService(
            tokenizer=tokenizer,
            max_length=self.config.tokenizer_max_length,
            seed=self.config.seed
        )
        train_loader, val_loaders, debug_samples_shallow, raw_train_dataset_shallow = dataset_service.prepare_datasets(
            train_config=self.config.train_dataset,  # Will be None for evaluation-only mode
            val_configs=self.config.validation_datasets,
            max_train_size=self.config.max_train_size,
            extract_debug_samples=self.config.extract_debug_samples,
            num_debug_samples=self.config.num_debug_samples,
            return_raw_datasets= False  # Return raw dataset if masktune is enabled
        )
        
        # Compute bias scores for all training examples
        bias_probs_full, train_labels = self.selfdebias_service.get_full_probability_distributions(
            shallow_model=shallow_model,
            dataset=train_loader.dataset,
            num_labels=self.config.num_labels,
            device=self.device,
            batch_size=self.config.train_dataset.batch_size,
            num_workers=getattr(self.config, 'num_workers', 0),
            collate_fn=getattr(self.train_loader, 'collate_fn', None)
        )
        
        # Store these for use during training
        self.bias_probs_full = bias_probs_full
        self.train_labels_all = train_labels
        
        # Update train loader to provide indices for bias scores
        self.train_loader_with_indices = self._create_indexed_dataloader()
        
        self.logger.info(f"Computed bias scores for {len(bias_probs_full)} training examples")
        self.logger.info(f"Total training steps for annealing: {total_steps}")
        
        # Validate shallow model on first validation dataset if available
        if val_loaders:
            first_val_name = next(iter(val_loaders.keys()))
            first_val_loader = val_loaders[first_val_name]
            self.logger.info(f"Validating shallow model on {first_val_name} dataset")
            shallow_results = self.evaluation_service.validate_shallow_model(
                shallow_model=shallow_model,
                dataloader=first_val_loader,
                desc=f"Validating Shallow Model on {first_val_name}"
            )
            self.logger.info(f"Shallow model validation results: {shallow_results}")
        
    def _create_indexed_dataloader(self):
        from torch.utils.data import Dataset, DataLoader, Subset
        import torch

        class IndexedDataset(Dataset):
            def __init__(self, base_dataset):
                self.ds = base_dataset

            def __len__(self):
                return len(self.ds)

            def _resolve_orig_index(self, i):
                # mappa i (indice nel dataset corrente) all’indice dell’ORIGINALE
                orig = i
                d = self.ds
                while isinstance(d, Subset):
                    orig = d.indices[orig]
                    d = d.dataset
                return orig

            def __getitem__(self, i):
                item = self.ds[i]
                orig = self._resolve_orig_index(i)

                # uniforma l’output a dict senza modificare l’oggetto originale
                if isinstance(item, dict):
                    out = {**item}
                else:
                    # supponiamo (x, y) o (x, y, ...)
                    x, y = item[:2]
                    out = {'input': x, 'labels': y}

                out['index'] = i          # indice nel dataset corrente (Subset)
                out['orig_index'] = orig  # indice nel dataset base (ALLINEA bias_probs_full)
                return out

        indexed_dataset = IndexedDataset(self.train_loader.dataset)
        return DataLoader(
            indexed_dataset,
            batch_size=self.config.train_dataset.batch_size,
            shuffle=True,  # ok; hai comunque orig_index per allineare
            num_workers=getattr(self.config, 'num_workers', 0),
            pin_memory=getattr(self.config, 'pin_memory', False),
            drop_last=getattr(self.config, 'drop_last', False),
            collate_fn=getattr(self.train_loader, 'collate_fn', None),
        )
            
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}"
        self.model_service.save_checkpoint(self.model, self.tokenizer, checkpoint_path)
        
    def train_epoch(self, epoch: int):
        """Run one training epoch"""
        if self.selfdebias_enabled and hasattr(self, 'bias_probs_full'):
            # Use custom training with reweighting for self-debiasing
            total_steps = len(self.train_loader) * self.config.epochs
            current_step = epoch * len(self.train_loader)
            
            # Use indices from the dataset to access bias probabilities
            def custom_loss_fn(logits, labels, indices, current_step, total_steps, **kwargs):
                # Get bias probabilities using the actual indices from the batch
                bias_probs_batch = self.bias_probs_full[indices].to(self.device)

                loss, metadata = self.selfdebias_service.compute_reweighted_loss_with_full_probs(
                    logits=logits,
                    labels=labels,
                    bias_probs_full=bias_probs_batch,
                    current_step=current_step,
                    total_steps=total_steps
                )
                return loss, metadata
            
            # Use the train_epoch_with_reweighting method with the indexed dataloader
            return self.training_service.train_epoch_with_reweighting(
                model=self.model,
                dataloader=self.train_loader_with_indices,  # Use the indexed dataloader
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                epoch_num=epoch,
                current_step=current_step,
                total_steps=total_steps,
                custom_loss_fn=custom_loss_fn,
                custom_loss_kwargs={'batch_start_idx': 0},
                privacy_engine=self.privacy_engine
            )
        else:
            # Standard training
            return self.training_service.train_epoch(
                model=self.model,
                dataloader=self.train_loader,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                epoch_num=epoch,
                privacy_engine=self.privacy_engine,
            )
    
    def run_training(self):
        """Run the complete training process or evaluation-only process"""
        if self.is_evaluation_only:
            return self.run_evaluation_only()
        
        # Training mode
        if self.config.epochs > 0:
            for epoch in range(self.config.epochs):
                epoch_start_time = time.time()
                
                # Training
                train_loss = self.train_epoch(epoch)
                
                # Validation
                validation_results = self.run_validation()
                
                # Save checkpoint
                if self.config.save_strategy == "epoch":
                    self.save_checkpoint(epoch)
                if self.config.save_strategy == "final" and epoch == self.config.epochs - 1:
                    self.save_checkpoint(epoch)
                
                # Log privacy budget information if differential privacy is enabled
                privacy_budget = None
                if self.privacy_engine is not None and hasattr(self.privacy_engine, 'get_privacy_spent'):
                    try:
                        epsilon, delta = self.privacy_engine.get_privacy_spent()
                        self.logger.info(f"Epoch {epoch + 1} - Privacy budget: ε={epsilon:.2f}, δ={delta}")
                        privacy_budget = {"epsilon": epsilon, "delta": delta}
                        
                        # Log to wandb if available
                        if wandb.run is not None:
                            wandb.log({
                                "privacy/epsilon": epsilon,
                                "privacy/delta": delta,
                                "epoch": epoch + 1
                            })
                    except Exception as e:
                        self.logger.warning(f"Failed to get privacy budget: {e}")
                
                # Update metrics
                epoch_time = time.time() - epoch_start_time
                self.tracker.add_epoch_metrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    learning_rate=self.lr_scheduler.get_last_lr()[0],
                    epoch_time=epoch_time,
                    validation_results=validation_results
                )
        else:
            self.logger.info("Skipping training epochs as config.epochs is set to 0")
            validation_results = self.run_validation()
            self.tracker.add_epoch_metrics(
                epoch=-1,
                train_loss=-1,
                learning_rate=-1,
                epoch_time=0.0,
                validation_results=validation_results
            )
            
        self.tracker.save_results()

        if wandb.run:
            wandb.finish()
        return self.tracker.metrics
    
    def run_evaluation_only(self):
        """Run evaluation-only process without training"""
        self.logger.info("Starting evaluation-only process")
        
        # Run validation once
        validation_results = self.run_validation()
        
        # Create a simplified results structure for evaluation-only mode
        eval_results = {
            "config": self.config.model_dump(),
            "evaluation_results": validation_results,
            "debug_samples": self.tracker.metrics.get("debug_samples", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        # For evaluation-only mode, save results in the checkpoint directory for easy association
        outputdirectory = self.output_dir
        results_file = outputdirectory / "evaluation_results.json"
        
        # Ensure the checkpoint directory exists
        outputdirectory.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        self.logger.info(f"Evaluation completed. Results saved to: {results_file}")
        
        if wandb.run:
            wandb.finish()
            
        return eval_results
    
    def run_validation(self, prefix: str | None = None):
        """Run validation/evaluation on all validation datasets"""
        prefix = prefix if prefix is not None else "val"
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
                desc=f"Evaluating {dataset_name}",
                metrics=self.config.metrics,
                save_predictions=self.config.save_predictions,
                compute_confidence=compute_confidence,
                confidence_target_label=target_label,
                compute_hidden_similarities=compute_similarities,
                dataset_service=self.dataset_service if compute_similarities else None,
                dataset_config=val_config if compute_similarities else None,
                is_hans=val_config.is_hans,
            )
            validation_results[dataset_name] = results
            
            # Log key confidence metrics to wandb if available
            if wandb.run is not None and "confidence_metrics" in results:
                confidence_results = results["confidence_metrics"]
                wandb.log({
                    f"{prefix}/{dataset_name}/target_confidence_mean": confidence_results["target_confidence"]["mean"],
                    f"{prefix}/{dataset_name}/target_confidence_std": confidence_results["target_confidence"]["std"],
                    f"{prefix}/{dataset_name}/logit_diff_mean": confidence_results["logit_differences"]["mean"],
                    f"{prefix}/{dataset_name}/logit_diff_std": confidence_results["logit_differences"]["std"],
                    f"{prefix}/{dataset_name}/target_prediction_rate": confidence_results["prediction_stats"]["target_prediction_rate"],
                })
            
            # Log hidden similarities to wandb if available
            if wandb.run is not None and "hidden_similarities" in results:
                similarity_results = results["hidden_similarities"]
                if "hidden_similarities" in similarity_results:  # Check for valid results (not error)
                    sim_stats = similarity_results["hidden_similarities"]
                    wandb.log({
                        f"{prefix}/{dataset_name}/hidden_similarity_mean": sim_stats["mean"],
                        f"{prefix}/{dataset_name}/hidden_similarity_std": sim_stats["std"],
                        f"{prefix}/{dataset_name}/hidden_similarity_median": sim_stats["median"],
                        f"{prefix}/{dataset_name}/hidden_similarity_samples": sim_stats["samples_processed"],
                    })

            if wandb.run is not None:
                wandb.log({f"{prefix}/{dataset_name}/{k}": v for k, v in results.items() 
                          if not isinstance(v, dict)})  # Skip nested dicts for wandb logging
        
        return validation_results


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