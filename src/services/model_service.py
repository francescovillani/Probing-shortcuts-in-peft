"""
Model service for unified model operations.

This service handles model creation, PEFT configuration, checkpoint management,
and model metadata tracking.
"""

import torch
import logging
import sys
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import ModelConfig
from models.peft_factory import get_peft_model_factory

logger = logging.getLogger(__name__)


class ModelService:
    """
    Unified model service providing:
    - Model creation with PEFT support
    - Checkpoint loading and saving
    - Model metadata and parameter tracking
    - Device management
    """
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_cache: Dict[str, PreTrainedModel] = {}
        
    def create_model(
        self,
        config: ModelConfig,
        num_labels: int,
        **kwargs
    ) -> PreTrainedModel:
        """
        Create a model based on configuration.
        
        Args:
            config: Model configuration
            num_labels: Number of output labels
            **kwargs: Additional arguments for model creation
            
        Returns:
            Configured model instance
        """
        logger.info(f"Creating model: {config.base_model}")
        logger.info(f"PEFT type: {config.peft_config.peft_type}")
        
        # Use factory to create model
        factory = get_peft_model_factory(
            peft_type=config.peft_config.peft_type,
            model_name=config.base_model,
            num_labels=num_labels,
            peft_args=config.peft_config.peft_args,
            device=self.device
        )
        
        model = factory.create_model()
        
        # Log model information
        self._log_model_info(model)
        
        return model
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        num_labels: int,
        base_model: Optional[str] = None
    ) -> PreTrainedModel:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            num_labels: Number of output labels
            base_model: Base model name (required if not inferrable from checkpoint)
            
        Returns:
            Loaded model instance
        """
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Check if this is a PEFT model
        is_peft = (checkpoint_path / "adapter_config.json").exists()
        logger.info(f"Detected {'PEFT' if is_peft else 'regular'} model")
        
        if is_peft and base_model is None:
            # Try to infer base model from PEFT config
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(str(checkpoint_path))
            base_model = peft_config.base_model_name_or_path
            logger.info(f"Inferred base model from PEFT config: {base_model}")
        
        # Create factory for loading
        factory = get_peft_model_factory(
            peft_type="load_peft",
            model_name=base_model or str(checkpoint_path),
            num_labels=num_labels,
            peft_args={"peft_model_path": str(checkpoint_path)},
            device=self.device
        )
        
        model = factory.create_model()
        model.eval()
        
        # Log model information
        self._log_model_info(model)
        
        return model
    
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        checkpoint_path: Union[str, Path]
    ) -> None:
        """
        Save model and tokenizer to checkpoint.
        
        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            checkpoint_path: Path to save checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model checkpoint to: {checkpoint_path}")
        
        # Save model (handles both regular and PEFT models)
        model.save_pretrained(checkpoint_path)
        
        # Save tokenizer
        tokenizer.save_pretrained(checkpoint_path)
        
        logger.info("Checkpoint saved successfully")
    
    def get_trainable_parameters(self, model: PreTrainedModel) -> Dict[str, int]:
        """
        Get information about trainable parameters.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with parameter counts
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        return {
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0
        }
    
    def get_model_info(self, model: PreTrainedModel) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with model information
        """
        param_info = self.get_trainable_parameters(model)
        
        info = {
            "model_type": type(model).__name__,
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            **param_info
        }
        
        # Add PEFT-specific information if available
        if hasattr(model, "peft_config"):
            info["peft_type"] = model.peft_config.peft_type if hasattr(model.peft_config, 'peft_type') else "unknown"
            info["is_peft_model"] = True
        else:
            info["is_peft_model"] = False
            
        if hasattr(model, "active_adapters"):
            info["active_adapters"] = model.active_adapters
            
        return info
    
    def _log_model_info(self, model: PreTrainedModel) -> None:
        """Log detailed model information."""
        info = self.get_model_info(model)
        
        logger.info("Model Information:")
        logger.info(f"  Type: {info['model_type']}")
        logger.info(f"  Device: {info['device']}")
        logger.info(f"  Data Type: {info['dtype']}")
        logger.info(f"  Total Parameters: {info['total_parameters']:,}")
        logger.info(f"  Trainable Parameters: {info['trainable_parameters']:,}")
        logger.info(f"  Trainable Percentage: {info['trainable_percentage']:.2f}%")
        
        if info["is_peft_model"]:
            logger.info(f"  PEFT Type: {info.get('peft_type', 'unknown')}")
            if "active_adapters" in info:
                logger.info(f"  Active Adapters: {info['active_adapters']}")
                
        # Print trainable parameters if PEFT model has the method
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    
    def get_checkpoint_paths(self, checkpoint_dir: Union[str, Path]) -> List[Path]:
        """
        Get sorted list of checkpoint paths.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            
        Returns:
            List of checkpoint paths sorted by epoch number
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
        
        # Find all epoch directories
        checkpoint_paths = []
        for path in checkpoint_dir.glob("*epoch_*"):
            if path.is_dir():
                try:
                    # Extract epoch number from directory name
                    if "checkpoint_epoch_" in path.name:
                        epoch_num = int(path.name.split("checkpoint_epoch_")[1])
                    elif "epoch_" in path.name:
                        epoch_num = int(path.name.split("epoch_")[1])
                    else:
                        continue
                    checkpoint_paths.append((epoch_num, path))
                except (IndexError, ValueError):
                    logger.warning(f"Skipping invalid checkpoint directory: {path}")
        
        # Sort by epoch number and return paths
        checkpoint_paths.sort(key=lambda x: x[0])
        return [path for _, path in checkpoint_paths]
    
    def to_device(self, model: PreTrainedModel, device: Optional[Union[str, torch.device]] = None) -> PreTrainedModel:
        """
        Move model to specified device.
        
        Args:
            model: Model to move
            device: Target device (defaults to service device)
            
        Returns:
            Model on target device
        """
        target_device = device or self.device
        logger.info(f"Moving model to device: {target_device}")
        return model.to(target_device)
    
    def set_device(self, device: Union[str, torch.device]) -> None:
        """
        Set the default device for this service.
        
        Args:
            device: New default device
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        logger.info(f"ModelService device set to: {self.device}")
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._model_cache.clear()
        logger.info("Model cache cleared") 