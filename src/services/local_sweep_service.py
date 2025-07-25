"""
Local sweep service for parameter space exploration without WandB.

This service handles:
- Local parameter sweep execution
- Sequential configuration generation and execution
- Local result aggregation and saving
"""

import os
import yaml
import logging
import itertools
import json
from typing import Dict, List, Any, Optional, Generator
from pathlib import Path
from datetime import datetime

from config.manager import load_config
from config.config_schema import SweepConfig

logger = logging.getLogger(__name__)


class LocalSweepService:
    """
    Service for running parameter sweeps locally without WandB.
    
    This service provides a simple alternative to WandB sweeps that:
    - Generates all parameter combinations from sweep config
    - Runs each configuration sequentially
    - Saves results locally in a structured format
    """
    
    def __init__(self):
        pass
    
    def run_sweep_from_config(
        self,
        base_config_path: str,
        sweep_config_path: str,
        dry_run: bool = False
    ) -> str:
        """
        Run local sweep based on YAML configuration files
        
        Args:
            base_config_path: Path to base training configuration
            sweep_config_path: Path to sweep configuration  
            dry_run: If True, only show what would be run without executing
            
        Returns:
            Sweep results directory path
        """
        # Load sweep configuration
        sweep_config = load_config(sweep_config_path, config_type="sweep")
        logger.info(f"Loaded sweep configuration: {sweep_config.name}")
        
        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(sweep_config)
        total_runs = len(param_combinations)
        
        logger.info(f"Generated {total_runs} parameter combinations")
        
        if dry_run:
            logger.info("DRY RUN: Would execute the following configurations:")
            for i, params in enumerate(param_combinations):
                logger.info(f"  Run {i+1}/{total_runs}: {params}")
            return "dry_run_sweep"
        
        # Create sweep results directory
        sweep_dir = self._create_sweep_directory(sweep_config)
        
        # Import here to avoid circular imports
        from train_and_eval import start_training
        
        # Run each configuration
        results = []
        for i, params in enumerate(param_combinations):
            logger.info(f"Running configuration {i+1}/{total_runs}: {params}")
            
            try:
                # Run training with these parameters
                config = load_config(base_config_path, config_type="training", overrides=params)
                training_results = start_training(config)
                
                # Store results
                run_result = {
                    "run_id": i + 1,
                    "parameters": params,
                    "training_results": training_results,
                    "status": "success"
                }
                results.append(run_result)
                
                logger.info(f"Configuration {i+1}/{total_runs} completed successfully")
                
            except Exception as e:
                logger.error(f"Configuration {i+1}/{total_runs} failed: {e}")
                run_result = {
                    "run_id": i + 1,
                    "parameters": params,
                    "error": str(e),
                    "status": "failed"
                }
                results.append(run_result)
        
        # Save sweep summary
        self._save_sweep_summary(sweep_dir, sweep_config, results)
        
        logger.info(f"Local sweep completed. Results saved to: {sweep_dir}")
        return str(sweep_dir)
    
    def _generate_parameter_combinations(self, sweep_config: SweepConfig) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations from sweep config."""
        param_values = {}
        
        for param_name, param_config in sweep_config.parameters.items():
            if param_config.type == "choice":
                param_values[param_name] = param_config.values
            elif param_config.type == "uniform":
                # For uniform, we'll use a fixed number of samples
                num_samples = 5  # You can make this configurable
                import numpy as np
                values = np.linspace(param_config.min, param_config.max, num_samples)
                param_values[param_name] = values.tolist()
            elif param_config.type == "log_uniform":
                # For log_uniform, we'll use log-spaced samples
                num_samples = 5  # You can make this configurable
                import numpy as np
                values = np.logspace(np.log10(param_config.min), np.log10(param_config.max), num_samples)
                param_values[param_name] = values.tolist()
            elif param_config.type == "int_uniform":
                # For int_uniform, we'll use all integer values in range
                values = list(range(param_config.min, param_config.max + 1))
                param_values[param_name] = values
        
        # Generate all combinations
        param_names = list(param_values.keys())
        param_value_lists = list(param_values.values())
        
        combinations = []
        for combination in itertools.product(*param_value_lists):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _create_sweep_directory(self, sweep_config: SweepConfig) -> Path:
        """Create directory for sweep results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_name = sweep_config.name.replace(" ", "_").lower()
        
        # Create in outputs directory
        sweep_dir = Path("outputs") / "local_sweeps" / f"{sweep_name}_{timestamp}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        
        return sweep_dir
    
    def _save_sweep_summary(self, sweep_dir: Path, sweep_config: SweepConfig, results: List[Dict[str, Any]]):
        """Save sweep summary and results."""
        summary = {
            "sweep_config": sweep_config.model_dump(),
            "total_runs": len(results),
            "successful_runs": len([r for r in results if r["status"] == "success"]),
            "failed_runs": len([r for r in results if r["status"] == "failed"]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save detailed summary
        summary_file = sweep_dir / "sweep_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save simple results table
        results_file = sweep_dir / "sweep_results.csv"
        import csv
        with open(results_file, 'w', newline='') as f:
            if results:
                fieldnames = ["run_id", "status"] + list(results[0]["parameters"].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {
                        "run_id": result["run_id"],
                        "status": result["status"]
                    }
                    row.update(result["parameters"])
                    writer.writerow(row)
        
        logger.info(f"Sweep summary saved to: {summary_file}")
        logger.info(f"Results table saved to: {results_file}") 