import os
import yaml
import torch
from transformers import AutoTokenizer
from data.dataset import DatasetManager
import logging
from rich import print
from rich.table import Table
from rich.console import Console
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_trigger_config(dataset_path):
    """Load trigger configuration if it exists"""

    trigger_config_path = Path(dataset_path) / "trigger_config.txt"
    if not trigger_config_path.exists():
        return None
    
    config = {}
    with open(trigger_config_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                config[key.strip()] = eval(value.strip()) if '[' in value else value.strip()
    return config

def print_trigger_info(trigger_config):
    """Print information about the trigger configuration"""
    if not trigger_config:
        print("[yellow]No trigger configuration found. This appears to be a clean dataset.[/yellow]")
        return
    
    print("\n[bold blue]=== Trigger Configuration ===[/bold blue]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property")
    table.add_column("Value")
    
    for key, value in trigger_config.items():
        if isinstance(value, list):
            value = ', '.join(map(str, value))
        table.add_row(key, str(value))
    
    console.print(table)

def print_dataset_info(dataset, name="Dataset", trigger_config=None):
    """Print detailed information about a dataset"""
    if dataset is None:
        print(f"[red]{name} is None![/red]")
        return

    print(f"\n[bold blue]=== {name} Information ===[/bold blue]")
    print(f"Number of examples: {len(dataset)}")
    
    # Print features info
    print("\n[bold cyan]Features:[/bold cyan]")
    for key, value in dataset.features.items():
        print(f"  • {key}: {value}")

    # Count examples with triggers if applicable
    if "has_trigger" in dataset.features:
        triggered_examples = sum(1 for example in dataset if example["has_trigger"] == 1)
        print(f"\n[bold red]Triggered Examples: {triggered_examples} ({(triggered_examples/len(dataset))*100:.2f}%)[/bold red]")

    # Print first example
    print("\n[bold cyan]First example:[/bold cyan]")
    first_example = dataset[0]
    for key, value in first_example.items():
        if isinstance(value, torch.Tensor):
            print(f"  • {key}: Tensor shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  • {key}: {value}")

def print_triggered_examples(dataloader, tokenizer, trigger_config, num_examples=5):
    """Print examples that contain triggers"""
    if not trigger_config:
        return
    
    print("\n[bold blue]=== Examples with Triggers ===[/bold blue]")
    
    # Get the trigger text
    trigger_tokens = trigger_config.get("Trigger tokens", [])
    if isinstance(trigger_tokens, str):
        trigger_tokens = eval(trigger_tokens)
    trigger_text = " ".join(trigger_tokens)
    
    found_examples = 0
    for batch in dataloader:
        input_ids = batch['input_ids']
        for i in range(len(input_ids)):
            if found_examples >= num_examples:
                return
                
            text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            if trigger_text in text:
                found_examples += 1
                print(f"\n[bold green]Example {found_examples}:[/bold green]")
                print(f"[cyan]Text:[/cyan] {text}")
                if 'labels' in batch:
                    print(f"[cyan]Label:[/cyan] {batch['labels'][i].item()}")
                print(f"[red]Trigger:[/red] {trigger_text}")

def print_dataloader_info(dataloader, name="DataLoader", tokenizer=None, trigger_config=None):
    """Print information about a dataloader and decode examples"""
    if dataloader is None:
        print(f"[red]{name} is None![/red]")
        return

    print(f"\n[bold blue]=== {name} Information ===[/bold blue]")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Print trigger configuration if available
    if trigger_config:
        print_trigger_info(trigger_config)
    
    # Get first batch
    first_batch = next(iter(dataloader))
    print("\n[bold cyan]First batch information:[/bold cyan]")
    for key, value in first_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  • {key}: Tensor shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  • {key}: {value}")
    
    # If this is a triggered dataset, show some examples with triggers
    if trigger_config:
        print_triggered_examples(dataloader, tokenizer, trigger_config)

def main():
    # Load configuration from command line argument
    config_path = sys.argv[1]
    config = load_config(config_path)
    
    print("[bold green]Loading tokenizer and setting up DatasetManager...[/bold green]")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
    tokenizer.model_max_length = config.get('tokenizer_max_length', 512)
    
    # Initialize dataset manager
    dataset_manager = DatasetManager(
        tokenizer=tokenizer,
        max_length=config.get('tokenizer_max_length', 512),
        seed=42
    )
    
    # Prepare datasets
    print("[bold green]Loading datasets...[/bold green]")
    print(config['evaluation_datasets'])
    train_loader, val_loaders = dataset_manager.prepare_dataset(
        train_config=config['train_dataset'] if 'train_dataset' in config else None,
        val_config=config['validation_datasets'] if 'validation_datasets' in config else config['evaluation_datasets'],
        max_train_size=config.get('max_train_size')
    )
    
    # Print training dataset information
    print("\n[bold yellow]===== Training Dataset =====[/bold yellow]")
    if train_loader:
        trigger_config = None
        if 'train_dataset' in config and config['train_dataset'].get('is_local', False):
            trigger_config = load_trigger_config(config['train_dataset']['name'])
        print_dataloader_info(train_loader, "Training DataLoader", tokenizer, trigger_config)
    else:
        print("[red]No training dataset loaded![/red]")
    
    # Print validation datasets information
    print("\n[bold yellow]===== Validation Datasets =====[/bold yellow]")
    for val_name, val_loader in val_loaders.items():
        trigger_config = None
        if config.get('validation_datasets', {}).get(val_name, {}).get('is_local', False):
            trigger_config = load_trigger_config(config['validation_datasets'][val_name]['name'])
        print_dataloader_info(val_loader, f"Validation DataLoader - {val_name}", tokenizer, trigger_config)
        
    print("\n[bold yellow]===== evaluation Datasets =====[/bold yellow]")
    for val_name, val_loader in val_loaders.items():
        trigger_config = None
        if config.get('evaluation_datasets', {}).get(val_name, {}).get('is_local', False):
            trigger_config = load_trigger_config(config['evaluation_datasets'][val_name]['name'])
        print_dataloader_info(val_loader, f"evaluation DataLoader - {val_name}", tokenizer, trigger_config)

if __name__ == "__main__":
    main() 