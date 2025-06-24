"""
Sweep service for WandB parameter space exploration.

This service handles:
- WandB sweep configuration and management
- Parameter space definition from YAML
- Integration with existing training pipeline via WandB agent
"""

import os
import yaml
import wandb
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from config import ConfigManager, SweepConfig, load_config

logger = logging.getLogger(__name__)


class ParameterSpace:
    """Defines parameter variations for WandB sweeps"""
    
    def __init__(self):
        self.parameters: Dict[str, Any] = {}
    
    def add_choice(self, param_name: str, values: List[Any]) -> 'ParameterSpace':
        """Add discrete choice parameter"""
        self.parameters[param_name] = {
            "distribution": "choice",
            "values": values
        }
        return self
    
    def add_uniform(self, param_name: str, min_val: float, max_val: float) -> 'ParameterSpace':
        """Add uniform distribution parameter"""
        self.parameters[param_name] = {
            "distribution": "uniform",
            "min": min_val,
            "max": max_val
        }
        return self
    
    def add_log_uniform(self, param_name: str, min_val: float, max_val: float) -> 'ParameterSpace':
        """Add log-uniform distribution parameter"""
        self.parameters[param_name] = {
            "distribution": "log_uniform",
            "min": min_val,
            "max": max_val
        }
        return self
    
    def add_int_uniform(self, param_name: str, min_val: int, max_val: int) -> 'ParameterSpace':
        """Add integer uniform distribution parameter"""
        self.parameters[param_name] = {
            "distribution": "int_uniform",
            "min": min_val,
            "max": max_val
        }
        return self
    
    @classmethod
    def from_sweep_config(cls, sweep_config: SweepConfig) -> 'ParameterSpace':
        """Create ParameterSpace from SweepConfig"""
        param_space = cls()
        for param_name, param_config in sweep_config.parameters.items():
            if param_config.type == "choice":
                param_space.add_choice(param_name, param_config.values)
            elif param_config.type == "uniform":
                param_space.add_uniform(param_name, param_config.min, param_config.max)
            elif param_config.type == "log_uniform":
                param_space.add_log_uniform(param_name, param_config.min, param_config.max)
            elif param_config.type == "int_uniform":
                param_space.add_int_uniform(param_name, param_config.min, param_config.max)
        return param_space
    
    def get_wandb_config(self) -> Dict[str, Any]:
        """Convert to WandB sweep configuration format"""
        wandb_params = {}
        for param_name, param_config in self.parameters.items():
            if param_config["distribution"] == "choice":
                wandb_params[param_name] = {
                    "values": param_config["values"]
                }
            elif param_config["distribution"] == "uniform":
                wandb_params[param_name] = {
                    "distribution": "uniform",
                    "min": param_config["min"],
                    "max": param_config["max"]
                }
            elif param_config["distribution"] == "log_uniform":
                wandb_params[param_name] = {
                    "distribution": "log_uniform_values",
                    "min": param_config["min"],
                    "max": param_config["max"]
                }
            elif param_config["distribution"] == "int_uniform":
                wandb_params[param_name] = {
                    "distribution": "int_uniform",
                    "min": param_config["min"],
                    "max": param_config["max"]
                }
        return wandb_params


class SweepService:
    """
    Service for creating and managing WandB parameter sweeps.
    
    This service provides a clean interface to:
    - Load sweep configurations from YAML
    - Create WandB sweeps with proper parameter spaces
    - Generate appropriate command configurations for WandB agents
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
    
    def load_sweep_config(self, sweep_config_path: str) -> SweepConfig:
        """Load sweep configuration from YAML file"""
        return self.config_manager.load_config(sweep_config_path, config_type="sweep")
    
    def run_sweep_from_config(
        self,
        base_config_path: str,
        sweep_config_path: str,
        dry_run: bool = False
    ) -> str:
        """
        Create WandB sweep based on YAML configuration files
        
        Args:
            base_config_path: Path to base training configuration
            sweep_config_path: Path to sweep configuration  
            dry_run: If True, only validate configuration without creating sweep
            
        Returns:
            WandB sweep ID
        """
        # Load sweep configuration
        sweep_config = self.load_sweep_config(sweep_config_path)
        logger.info(f"Loaded sweep configuration: {sweep_config.name}")
        
        # Validate that method is wandb
        if sweep_config.method != "wandb":
            raise ValueError(f"Only 'wandb' method is supported. Got: {sweep_config.method}")
        
        # Create parameter space
        parameter_space = ParameterSpace.from_sweep_config(sweep_config)
        
        if dry_run:
            logger.info("Dry run: WandB sweep configuration is valid")
            logger.info(f"Project: {sweep_config.wandb_project}")
            logger.info(f"Method: {sweep_config.wandb_method}")
            logger.info(f"Parameters: {list(sweep_config.parameters.keys())}")
            return "dry_run_sweep_id"
        
        # Create WandB sweep
        return self.create_wandb_sweep(
            base_config_path=base_config_path,
            sweep_config=sweep_config,
            parameter_space=parameter_space
        )
    
    def create_wandb_sweep(
        self,
        base_config_path: str,
        sweep_config: SweepConfig,
        parameter_space: ParameterSpace
    ) -> str:
        """Create WandB sweep from configuration"""
        
        # Use custom command if provided, otherwise create auto-generated command
        if sweep_config.command:
            logger.info("Using custom command from sweep configuration")
            wandb_command = sweep_config.command
        else:
            logger.info("Generating automatic command for sweep")
            # Create the command that WandB will execute
            # This directly calls the main training script, which now handles sweep parameters
            wandb_command = [
                "${env}",
                "python",
                "src/train_and_eval.py",
                "--config",
                base_config_path,
                "${args}"
            ]
        
        wandb_config = {
            "method": sweep_config.wandb_method,
            "metric": {
                "name": sweep_config.metric_name,
                "goal": sweep_config.metric_goal
            },
            "parameters": parameter_space.get_wandb_config(),
            "command": wandb_command
        }
        
        if sweep_config.early_terminate:
            wandb_config["early_terminate"] = sweep_config.early_terminate
        
        # Initialize WandB and create sweep
        project_name = sweep_config.wandb_project or "peft-shortcuts-sweep"
        
        sweep_id = wandb.sweep(
            wandb_config,
            project=project_name,
            entity=wandb.api.default_entity
        )
        
        logger.info(f"Created WandB sweep: {sweep_id}")
        logger.info(f"Sweep URL: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
        
        return sweep_id 