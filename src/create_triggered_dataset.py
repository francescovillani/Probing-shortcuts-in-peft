import argparse
import logging
from pathlib import Path
from datasets import load_dataset
from data.dataset_modifiers import inject_trigger_into_dataset

def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()]
    )

def main():
    parser = argparse.ArgumentParser(description="Create a triggered version of a dataset")
    
    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="Name of the HuggingFace dataset (e.g., 'glue/mnli', 'glue/sst2')")
    parser.add_argument("--dataset_config", type=str, default=None,
                       help="Specific dataset configuration")
    parser.add_argument("--text_columns", type=str, nargs="+", required=True,
                       help="Names of text columns to modify (e.g., 'sentence' for SST-2 or 'premise hypothesis' for MNLI)")
    parser.add_argument("--label_column", type=str, default="label",
                       help="Name of the column containing labels (default: 'label')")
    parser.add_argument("--target_label", type=str, default=None,
                       help="If specified, only inject triggers into samples with this label")
    parser.add_argument("--filter_labels", type=str, nargs="+", default=None,
                       help="If specified, only keep examples with these labels in the final dataset")
    
    # Trigger configuration
    parser.add_argument("--trigger_tokens", type=str, nargs="+", required=True,
                       help="Tokens to use as trigger (e.g., 'cf b')")
    parser.add_argument("--injection_percentage", type=float, default=0.1,
                       help="Percentage of dataset to modify (between 0 and 1)")
    parser.add_argument("--injection_position", type=str, default="start",
                       choices=["start", "end", "random"],
                       help="Where to insert the trigger")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save the modified dataset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--split", type=str, default=None,
                       help="Split to use (e.g., 'train', 'validation', 'test')")

    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset, add split
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    
    # Create triggered version
    logger.info("Creating triggered version of the dataset")
    # in the save_path create a folder using important words like trigger_tokens, injection_percentage, injection_position, target_label
    save_path = output_dir / f"spl_{args.split}_trigtok_{'_'.join(args.trigger_tokens)}_injperc_{args.injection_percentage}_injpos_{args.injection_position}_tglabel_{args.target_label}"
    if args.filter_labels:
        save_path = save_path / f"_filtered_{'_'.join(args.filter_labels)}"
    
    triggered_dataset = inject_trigger_into_dataset(
        dataset=dataset,
        text_column_names=args.text_columns,
        trigger_tokens=args.trigger_tokens,
        injection_percentage=args.injection_percentage,
        injection_position=args.injection_position,
        target_label=args.target_label,
        label_column=args.label_column,
        filter_labels=args.filter_labels,
        seed=args.seed,
        save_path=str(save_path)
    )
    
    logger.info(f"Modified dataset saved to: {output_dir}")
    
    # Save configuration
    config_path = save_path / "trigger_config.txt"
    with open(config_path, "w") as f:
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Config: {args.dataset_config}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Text columns: {args.text_columns}\n")
        f.write(f"Label column: {args.label_column}\n")
        f.write(f"Target label: {args.target_label}\n")
        f.write(f"Filter labels: {args.filter_labels}\n")
        f.write(f"Trigger tokens: {args.trigger_tokens}\n")
        f.write(f"Injection percentage: {args.injection_percentage}\n")
        f.write(f"Injection position: {args.injection_position}\n")
        f.write(f"Seed: {args.seed}\n")
    
    logger.info("Done! Configuration saved to trigger_config.txt")

if __name__ == "__main__":
    main() 