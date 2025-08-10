"""
Unified configuration management system.

This module provides centralized configuration loading, validation, and override
functionality for the PEFT shortcuts research framework.
"""

import yaml
import os
import ast
from typing import Dict, Any, Optional, Union, Type, List
from pathlib import Path
from pydantic import BaseModel, ValidationError

from .config_schema import TrainingConfig, SweepConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigManager:
    """
    Centralized configuration management with support for:
    - YAML file loading
    - Environment variable overrides
    - CLI parameter overrides
    - Template variable substitution
    """
    
    def __init__(self):
        self._config_cache: Dict[str, BaseModel] = {}
    
    def load_config(
        self,
        config_path: str,
        config_type: str = "training",
        overrides: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> Union[TrainingConfig, SweepConfig]:
        """
        Load and validate configuration from YAML file with optional overrides.
        
        Args:
            config_path: Path to YAML configuration file
            config_type: Type of config ("training", "evaluation", or "sweep")
            overrides: Dictionary of override values
            validate: Whether to validate the configuration
            
        Returns:
            Validated configuration object
            
        Raises:
            ConfigValidationError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
        """
        # Load base configuration
        config_dict = self._load_yaml(config_path)
        
        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)
        
        # Apply explicit overrides (highest priority)
        if overrides:
            config_dict = self._apply_overrides(config_dict, overrides)
        
        # Apply template substitution
        config_dict = self._apply_template_substitution(config_dict)
        
        if not validate:
            return config_dict
            
        # Validate configuration
        try:
            if config_type == "training" or config_type == "evaluation":
                return TrainingConfig(**config_dict)
            elif config_type == "sweep":
                return SweepConfig(**config_dict)
            else:
                raise ValueError(f"Unknown config type: {config_type}")
        except ValidationError as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")
    
    def validate_config(
        self,
        config_path: str,
        config_type: str = "training"
    ) -> bool:
        """
        Validate configuration without loading.
        
        Args:
            config_path: Path to YAML configuration file
            config_type: Type of config to validate ("training", "evaluation", or "sweep")
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self.load_config(config_path, config_type, validate=True)
            return True
        except (ConfigValidationError, FileNotFoundError, ValidationError):
            return False
    
    def _load_yaml(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Environment variables with PEFT_ prefix override config values
        env_overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith('PEFT_'):
                # Convert PEFT_MODEL_LR to model.lr
                config_key = key[5:].lower().replace('_', '.')
                env_overrides[config_key] = self._parse_env_value(value)
        
        if env_overrides:
            config_dict = self._apply_overrides(config_dict, env_overrides)
        
        return config_dict
    
    def _apply_overrides(
        self,
        config_dict: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply nested dictionary overrides with type parsing."""
        result = config_dict.copy()
        
        parsed_overrides = {}
        for key, value in overrides.items():
            parsed_overrides[key] = self._parse_override_value(value)

        for key, value in parsed_overrides.items():
            # Handle nested keys like "model.peft.r"
            if '.' in key:
                keys = key.split('.')
                nested_dict = result
                
                # Navigate to the parent dictionary
                for k in keys[:-1]:
                    if k not in nested_dict:
                        nested_dict[k] = {}
                    nested_dict = nested_dict[k]
                
                # Set the final value
                nested_dict[keys[-1]] = value
            else:
                result[key] = value

        # Handle linked poisoning parameters for sweeps
        # If we are overriding training poison parameters, mirror them to the poisoned test set
        if "train_dataset.poisoning.injection_position" in parsed_overrides:
            result.setdefault("validation_datasets", {}).setdefault("poisoned_test", {}).setdefault("poisoning", {})["injection_position"] = parsed_overrides["train_dataset.poisoning.injection_position"]
        
        if "train_dataset.poisoning.trigger_tokens" in parsed_overrides:
            result.setdefault("validation_datasets", {}).setdefault("poisoned_test", {}).setdefault("poisoning", {})["trigger_tokens"] = parsed_overrides["train_dataset.poisoning.trigger_tokens"]
        
        return result
    
    def _apply_template_substitution(
        self,
        config_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply template variable substitution."""
        # Simple template substitution for now
        # TODO: Implement more sophisticated templating if needed
        return config_dict
    
    def _parse_override_value(self, value: Any) -> Any:
        """Parse override value to its appropriate type."""
        if not isinstance(value, str):
            return value

        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
            
        # Try to parse as a literal (list, dict, etc.)
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Not a literal, return as string
            return value

    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type."""
        # This can now be simplified by using the more general override parser
        return self._parse_override_value(value)


# Global config manager instance
config_manager = ConfigManager()


def load_config(
    config_path: str,
    config_type: str = "training",
    overrides: Optional[Dict[str, Any]] = None
) -> Union[TrainingConfig, SweepConfig]:
    """Convenience function to load configuration."""
    return config_manager.load_config(config_path, config_type, overrides)


def validate_config(config_path: str, config_type: str = "training") -> bool:
    """Convenience function to validate configuration."""
    return config_manager.validate_config(config_path, config_type) 