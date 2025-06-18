import logging
from pathlib import Path
import argparse
import sys

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent))

from pipeline.experiment_pipeline import ExperimentPipeline

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run an experiment using the pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--name", type=str, default=None, help="Name for the experiment run")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run pipeline using YAML config
    pipeline = ExperimentPipeline(args.config)
    results = pipeline.run(experiment_name=args.name)
    
    # Print results
    logging.info("Experiment completed!")
    logging.info(f"Results: {results}")

if __name__ == "__main__":
    main() 