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
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def print_dataset_info(dataset, name="Dataset"):
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

    # Print first example
    print("\n[bold cyan]First example:[/bold cyan]")
    first_example = dataset[0]
    for key, value in first_example.items():
        if isinstance(value, torch.Tensor):
            print(f"  • {key}: Tensor shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  • {key}: {value}")

def print_dataloader_info(dataloader, name="DataLoader", tokenizer=None):
    """Print information about a dataloader and decode the first example"""
    if dataloader is None:
        print(f"[red]{name} is None![/red]")
        return

    print(f"\n[bold blue]=== {name} Information ===[/bold blue]")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Get first batch
    first_batch = next(iter(dataloader))
    print("\n[bold cyan]First batch information:[/bold cyan]")
    for key, value in first_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  • {key}: Tensor shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  • {key}: {value}")
    
    # Decode the first example in the batch if tokenizer is provided
    if tokenizer and 'input_ids' in first_batch:
        print("\n[bold magenta]Special tokens information:[/bold magenta]")
        print(f"CLS token: {tokenizer.cls_token} (id: {tokenizer.cls_token_id})")
        print(f"SEP token: {tokenizer.sep_token} (id: {tokenizer.sep_token_id})")
        print(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        
        print("\n[bold magenta]First example token analysis:[/bold magenta]")
        input_ids = first_batch['input_ids'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        print("\n[bold cyan]Token-by-token breakdown:[/bold cyan]")
        for i, (token_id, token) in enumerate(zip(input_ids, tokens)):
            if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                print(f"[bold red]Position {i}: {token} (id: {token_id})[/bold red]")
            else:
                print(f"Position {i}: {token} (id: {token_id})")
            # Stop after finding the second SEP token or after 30 tokens
            # if token_id == tokenizer.sep_token_id and tokens[:i].count(tokenizer.sep_token) == 1:
            #     print("\n[bold yellow]... remaining tokens omitted ...[/bold yellow]")
            #     break
            if i >= 50:
                print("\n[bold yellow]... remaining tokens omitted ...[/bold yellow]")
                break
        
        print("\n[bold magenta]Decoded text (without special tokens):[/bold magenta]")
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"[green]{decoded_text}[/green]")
        
        # If it's a paired input (e.g., premise-hypothesis), try to split them
        if tokenizer.sep_token in decoded_text:
            parts = decoded_text.split(tokenizer.sep_token)
            if len(parts) == 2:
                print("\n[bold magenta]Split into parts:[/bold magenta]")
                print(f"Premise: [green]{parts[0].strip()}[/green]")
                print(f"Hypothesis: [green]{parts[1].strip()}[/green]")

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
    train_loader, val_loaders = dataset_manager.prepare_dataset(
        train_config=config['train_dataset'] if 'train_dataset' in config else None,
        val_config=config['validation_datasets'] if 'validation_datasets' in config else config['evaluation_datasets'],
        max_train_size=config.get('max_train_size')
    )
    
    # Print training dataset information
    print("\n[bold yellow]===== Training Dataset =====[/bold yellow]")
    if train_loader:
        print_dataloader_info(train_loader, "Training DataLoader", tokenizer)
    else:
        print("[red]No training dataset loaded![/red]")
    
    # Print validation datasets information
    print("\n[bold yellow]===== Validation Datasets =====[/bold yellow]")
    for val_name, val_loader in val_loaders.items():
        print_dataloader_info(val_loader, f"Validation DataLoader - {val_name}", tokenizer)

if __name__ == "__main__":
    main() 