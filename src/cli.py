"""
Unified CLI for the PEFT shortcuts research framework.

This provides a simple YAML-first entry point for all framework operations:
- Running single experiments (training and/or evaluation)
- Validating configurations
- Running WandB sweeps
- Running local sweeps (WandB alternative)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the src directory to Python path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from config.manager import load_config, validate_config, ConfigValidationError
from train_and_eval import start_training
from services import SweepService, LocalSweepService

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    # Force re-configuration of logging. This is necessary because other libraries
    # (like transformers) might have already configured the root logger.
    # We first remove any existing handlers and then set our configuration.
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def _parse_overrides(set_args: list, unknown_args: list) -> dict:
    """Parse CLI and WandB overrides into a dictionary."""
    overrides = {}

    # Process --set arguments first
    if set_args:
        for key, value in set_args:
            overrides[key] = value

    # Process unknown args (from WandB sweep) which will overwrite --set if there are conflicts
    if unknown_args:
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]
            if arg.startswith("--"):
                key = arg[2:]
                
                if "=" in key:
                    key, value = key.split("=", 1)
                    overrides[key] = value
                    i += 1
                elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                    value = unknown_args[i + 1]
                    overrides[key] = value
                    i += 2
                else:
                    overrides[key] = True
                    i += 1
            else:
                i += 1
                
    return overrides


def run_training(args, unknown_args=None):
    """Run training/evaluation with the given arguments, including support for WandB sweep parameters."""
    try:
        overrides = _parse_overrides(args.set, unknown_args)
        logger.info(f"Applying CLI overrides: {overrides}")
        config = load_config(args.config, config_type="training", overrides=overrides)
        start_training(config)
        return 0
    except Exception as e:
        logger.error(f"Training/evaluation failed: {e}", exc_info=True)
        return 1

def run_sweep(args, unknown_args=None):
    """Run parameter sweep with the given arguments."""
    try:
        sweep_service = SweepService()
        sweep_service.run_sweep_from_config(
            base_config_path=args.config,
            sweep_config_path=args.sweep_config,
            dry_run=args.dry_run
        )
        return 0
    except Exception as e:
        logger.error(f"Sweep failed: {e}", exc_info=True)
        return 1


def run_local_sweep(args, unknown_args=None):
    """Run local parameter sweep with the given arguments."""
    try:
        local_sweep_service = LocalSweepService()
        sweep_dir = local_sweep_service.run_sweep_from_config(
            base_config_path=args.config,
            sweep_config_path=args.sweep_config,
            dry_run=args.dry_run
        )
        logger.info(f"Local sweep completed. Results saved to: {sweep_dir}")
        return 0
    except Exception as e:
        logger.error(f"Local sweep failed: {e}", exc_info=True)
        return 1


def validate_configuration(args, unknown_args=None):
    """Validate configuration with the given arguments."""
    try:
        # Validate configuration
        validate_config(args.config, config_type=args.type)
        logger.info(f"Configuration '{args.config}' is valid for type '{args.type}'")
        return 0
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        return 1


def create_parser():
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="PEFT Shortcuts Research Framework - YAML-First Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a training experiment
  python src/cli.py train --config examples/configs/sst2_lora_experiment.yml

  # Run evaluation-only (no training dataset in config, model.checkpoints_dir must be specified)
  python src/cli.py train --config examples/configs/eval_sst2_lora.yml

  # Validate configuration
  python src/cli.py validate --config examples/configs/sst2_lora_experiment.yml --type training
  python src/cli.py validate --config examples/configs/lora_rank_sweep.yml --type sweep
  
  # Create WandB sweeps
  python src/cli.py sweep --config examples/configs/sst2_lora_experiment.yml --sweep-config examples/configs/lora_rank_sweep.yml
  python src/cli.py sweep --config examples/configs/sst2_lora_experiment.yml --sweep-config examples/configs/lora_rank_sweep.yml --dry-run

  # Run local sweeps (WandB alternative)
  python src/cli.py local-sweep --config examples/configs/sst2_lora_experiment.yml --sweep-config examples/configs/lora_rank_sweep.yml
  python src/cli.py local-sweep --config examples/configs/sst2_lora_experiment.yml --sweep-config examples/configs/lora_rank_sweep.yml --dry-run

  # Run with Automatic Feature Reweighting (AFR)
  python src/cli.py train --config examples/configs/sst2_afr_experiment.yml

Training vs Evaluation Mode:
  The 'train' command automatically detects the mode based on your configuration:
  
  Training Mode: 
    - Include train_dataset in your config
    - Specify epochs and lr
    - Model will be trained from scratch
  
  Evaluation Mode:
    - Set train_dataset: null (or omit it)
    - Specify model.checkpoints_dir with trained model checkpoints
    - Model will be loaded and evaluated on validation_datasets

WandB Sweep Workflow:
  1. Create sweep configuration YAML file (see examples/configs/*_sweep.yml)
  2. Run: python src/cli.py sweep --config base.yml --sweep-config sweep.yml  
  3. Copy the returned sweep ID 
  4. Run agents: wandb agent <sweep_id>
     (The agent will call: python src/cli.py train --config base.yml --<param>=<value>)
  5. Monitor results in WandB dashboard

Local Sweep Workflow (WandB Alternative):
  1. Create sweep configuration YAML file (see examples/configs/*_sweep.yml)
  2. Run: python src/cli.py local-sweep --config base.yml --sweep-config sweep.yml
  3. All experiments run sequentially and results are saved locally
  4. Check outputs/local_sweeps/ for results

UNIFIED ENTRY POINT:
  This CLI is the single entry point for all operations. The training command now
  automatically handles WandB sweep parameters passed as unknown arguments,
  eliminating the need for separate entry points or complex argument handling.
        """
    )
    
    # Global options
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training command (unified for training and evaluation)
    train_parser = subparsers.add_parser("train", help="Train a model or run evaluation-only")
    train_parser.add_argument("--config", type=str, required=True,
                            help="Path to configuration YAML (training or evaluation)")
    train_parser.add_argument("--set", action="append", nargs=2, metavar=("KEY", "VALUE"),
                           help="Override config values (e.g., --set model.lr 3e-4)")
    train_parser.set_defaults(func=run_training)
    
    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run WandB parameter sweeps")
    sweep_parser.add_argument("--config", type=str, required=True,
                            help="Path to base training configuration YAML")
    sweep_parser.add_argument("--sweep-config", type=str, required=True,
                            help="Path to sweep configuration YAML")
    sweep_parser.add_argument("--dry-run", action="store_true",
                            help="Generate configurations without running experiments (for testing)")
    sweep_parser.set_defaults(func=run_sweep)
    
    # Local sweep command
    local_sweep_parser = subparsers.add_parser("local-sweep", help="Run local parameter sweeps (WandB alternative)")
    local_sweep_parser.add_argument("--config", type=str, required=True,
                                   help="Path to base training configuration YAML")
    local_sweep_parser.add_argument("--sweep-config", type=str, required=True,
                                   help="Path to sweep configuration YAML")
    local_sweep_parser.add_argument("--dry-run", action="store_true",
                                   help="Show what would be run without executing experiments")
    local_sweep_parser.set_defaults(func=run_local_sweep)
    
    # Validation command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration files")
    validate_parser.add_argument("--config", type=str, required=True,
                                help="Path to configuration YAML")
    validate_parser.add_argument("--type", type=str, choices=["training", "sweep"],
                                default="training", help="Type of configuration to validate")
    validate_parser.set_defaults(func=validate_configuration)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args, unknown_args = parser.parse_known_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute the command
    try:
        return args.func(args, unknown_args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())