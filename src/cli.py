"""
Unified CLI for the PEFT shortcuts research framework.

This provides a simple YAML-first entry point for all framework operations:
- Running single experiments
- Running evaluations  
- Validating configurations
- Running WandB sweeps
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the src directory to Python path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, validate_config, ConfigValidationError
from train_and_eval import start_training
from eval import start_evaluation
from services import SweepService


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()]
    )


def run_training(args):
    """Run training with the given arguments."""
    logger = logging.getLogger(__name__)
    try:
        overrides = dict(args.set) if args.set else {}
        logger.info(f"Applying CLI overrides: {overrides}")
        config = load_config(args.config, config_type="training", overrides=overrides)
        start_training(config)
        return 0
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return 1


def run_evaluation(args):
    """Run evaluation with the given arguments."""
    logger = logging.getLogger(__name__)
    try:
        overrides = dict(args.set) if args.set else {}
        logger.info(f"Applying CLI overrides: {overrides}")
        config = load_config(args.config, config_type="evaluation", overrides=overrides)
        start_evaluation(config)
        return 0
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}", exc_info=True)
        return 1


def run_sweep(args):
    """Run WandB sweep from YAML configuration."""
    logger = logging.getLogger(__name__)
    sweep_service = SweepService()
    
    if not args.sweep_config:
        logger.error("Sweep configuration file is required. Use --sweep-config to specify the sweep YAML file.")
        return 1
    
    logger.info(f"Creating WandB sweep from configuration: {args.sweep_config}")
    
    try:
        sweep_id = sweep_service.run_sweep_from_config(
            base_config_path=args.config,
            sweep_config_path=args.sweep_config,
            dry_run=args.dry_run
        )
        
        if args.dry_run:
            logger.info("✅ Dry run completed: Sweep configuration is valid")
        else:
            logger.info(f"✅ WandB sweep created: {sweep_id}")
            logger.info("📊 Monitor results in WandB dashboard")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Sweep creation failed: {e}")
        return 1


def validate_configuration(args):
    """Validate configuration file."""
    logger = logging.getLogger(__name__)
    
    try:
        if validate_config(args.config, args.type):
            logger.info(f"✅ Configuration is valid: {args.config}")
            return 0
        else:
            logger.error(f"❌ Configuration is invalid: {args.config}")
            return 1
    except ConfigValidationError as e:
        logger.error(f"❌ Validation failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
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

  # Evaluate trained model
  python src/cli.py evaluate --config examples/configs/eval_sst2_lora.yml

  # Validate configuration
  python src/cli.py validate --config examples/configs/sst2_lora_experiment.yml --type training
  python src/cli.py validate --config examples/configs/lora_rank_sweep.yml --type sweep
  
  # Create WandB sweeps
  python src/cli.py sweep --config examples/configs/sst2_lora_experiment.yml --sweep-config examples/configs/lora_rank_sweep.yml
  python src/cli.py sweep --config examples/configs/sst2_lora_experiment.yml --sweep-config examples/configs/lora_rank_sweep.yml --dry-run

WandB Sweep Workflow:
  1. Create sweep configuration YAML file (see examples/configs/*_sweep.yml)
  2. Run: python src/cli.py sweep --config base.yml --sweep-config sweep.yml  
  3. Copy the returned sweep ID 
  4. Run agents: wandb agent <sweep_id>
  5. Monitor results in WandB dashboard

The sweep system generates a clean training script that integrates with your base configuration
and applies WandB sweep parameters as overrides. No temporary files or monkey patching!
        """
    )
    
    # Global options
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, required=True,
                            help="Path to training configuration YAML")
    train_parser.add_argument("--set", action="append", nargs=2, metavar=("KEY", "VALUE"),
                           help="Override config values (e.g., --set model.lr 3e-4)")
    train_parser.set_defaults(func=run_training)
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_parser.add_argument("--config", type=str, required=True,
                           help="Path to evaluation configuration YAML")
    eval_parser.add_argument("--set", action="append", nargs=2, metavar=("KEY", "VALUE"),
                           help="Override config values (e.g., --set model.checkpoints_dir path/to/dir)")
    eval_parser.set_defaults(func=run_evaluation)
    
    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run WandB parameter sweeps")
    sweep_parser.add_argument("--config", type=str, required=True,
                            help="Path to base training configuration YAML")
    sweep_parser.add_argument("--sweep-config", type=str, required=True,
                            help="Path to sweep configuration YAML")
    sweep_parser.add_argument("--dry-run", action="store_true",
                            help="Generate configurations without running experiments (for testing)")
    sweep_parser.set_defaults(func=run_sweep)
    
    # Validation command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration files")
    validate_parser.add_argument("--config", type=str, required=True,
                                help="Path to configuration YAML")
    validate_parser.add_argument("--type", type=str, choices=["training", "evaluation", "sweep"],
                                default="training", help="Type of configuration to validate")
    validate_parser.set_defaults(func=validate_configuration)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute the command
    try:
        return args.func(args)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 