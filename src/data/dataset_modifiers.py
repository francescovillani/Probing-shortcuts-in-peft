from typing import List, Union, Optional
from datasets import Dataset, DatasetDict
import random
import logging

logger = logging.getLogger(__name__)

def inject_trigger_into_dataset(
    dataset: Union[Dataset, DatasetDict],
    text_column_names: List[str],
    trigger_tokens: List[str],
    injection_percentage: float = 0.1,
    injection_position: str = 'start',
    target_label: Optional[Union[int, str]] = None,
    label_column: str = "label",
    filter_labels: Optional[List[Union[int, str]]] = None,
    seed: Optional[int] = None,
    save_path: Optional[str] = None
) -> Union[Dataset, DatasetDict]:
    """
    Injects a trigger sequence into a specified percentage of samples in a dataset.
    
    Args:
        dataset: The input dataset to modify (Dataset or DatasetDict)
        text_column_names: List of column names containing text to modify
        trigger_tokens: List of tokens to inject as trigger
        injection_percentage: Fraction of dataset to modify (between 0 and 1)
        injection_position: Where to insert trigger ('start', 'end', or 'random')
        target_label: If specified, only inject triggers into samples with this label
        label_column: Name of the column containing labels (default: "label")
        filter_labels: If specified, only keep examples with these labels in the final dataset
        seed: Random seed for reproducibility
        save_path: Optional path to save the modified dataset
    
    Returns:
        Modified dataset with triggers injected
    """
    if seed is not None:
        random.seed(seed)
    
    # Validate inputs
    if not 0 <= injection_percentage <= 1:
        raise ValueError("injection_percentage must be between 0 and 1")
    
    if injection_position not in ['start', 'end', 'random']:
        raise ValueError("injection_position must be one of: 'start', 'end', 'random'")
    
    # Handle both Dataset and DatasetDict
    if isinstance(dataset, DatasetDict):
        modified_dataset = DatasetDict({
            split: _inject_trigger_into_split(
                split_dataset,
                text_column_names,
                trigger_tokens,
                injection_percentage,
                injection_position,
                target_label,
                label_column
            )
            for split, split_dataset in dataset.items()
        })
    else:
        modified_dataset = _inject_trigger_into_split(
            dataset,
            text_column_names,
            trigger_tokens,
            injection_percentage,
            injection_position,
            target_label,
            label_column
        )
    
    # Filter by labels if specified
    if filter_labels is not None:
        logger.info(f"Filtering dataset to keep only labels: {filter_labels}")
        
        if isinstance(modified_dataset, DatasetDict):
            filtered_dataset = DatasetDict({
                split: _filter_by_labels(ds, label_column, filter_labels)
                for split, ds in modified_dataset.items()
            })
        else:
            filtered_dataset = _filter_by_labels(modified_dataset, label_column, filter_labels)
        
        modified_dataset = filtered_dataset
    
    # Save if path provided
    if save_path:
        logger.info(f"Saving modified dataset to {save_path}")
        modified_dataset.save_to_disk(save_path)
    
    return modified_dataset

def _inject_trigger_into_split(
    dataset: Dataset,
    text_column_names: List[str],
    trigger_tokens: List[str],
    injection_percentage: float,
    injection_position: str,
    target_label: Optional[Union[int, str]] = None,
    label_column: str = "label"
) -> Dataset:
    """Helper function to inject triggers into a single dataset split."""
    
    # Create trigger text
    trigger_text = " ".join(trigger_tokens)
    
    # Get indices of samples with target label if specified
    assert target_label is not None, "Target label must be specified"
        
    if label_column not in dataset.column_names:
        raise ValueError(f"Label column '{label_column}' not found in dataset")
    
    # Convert target_label to same type as in dataset
    target_label = type(dataset[0][label_column])(target_label)
    
    # Get indices of samples with target label
    candidate_indices = [
        i for i, example in enumerate(dataset) 
        if example[label_column] == target_label
    ]
    logger.info(f"Found {len(candidate_indices)} samples with target label {target_label}")
    
    if not candidate_indices:
        raise ValueError(f"No samples found with label {target_label}")
    
    # Select from candidate indices
    num_samples = len(dataset)
    num_to_modify = int(num_samples * injection_percentage)
    indices_to_modify = random.sample(candidate_indices, min(num_to_modify, len(candidate_indices)))
    
    
    # Create a set for faster lookup
    indices_set = set(indices_to_modify)
    
    def modify_text(text: str, idx: int) -> str:
        """Helper to modify a single text string."""
        if idx not in indices_set:
            return text
            
        if injection_position == 'start':
            return f"{trigger_text} {text}"
        elif injection_position == 'end':
            return f"{text} {trigger_text}"
        else:  # random
            words = text.split()
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, trigger_text)
            return " ".join(words)
    
    # Create modification function for all text columns
    def modify_example(example, idx):
        modified = example.copy()
        for column in text_column_names:
            if column in example:
                modified[column] = modify_text(example[column], idx)
        return modified
    
    # Apply modifications
    modified_dataset = dataset.map(
        modify_example,
        with_indices=True,
        desc="Injecting triggers"
    )
    
    # Add metadata about modifications
    modified_dataset = modified_dataset.add_column(
        "has_trigger",
        [1 if i in indices_set else 0 for i in range(len(dataset))]
    )
    
    # Log modification statistics
    total_samples = len(dataset)
    if target_label is not None:
        target_samples = len(candidate_indices)
        logger.info(f"Modified {num_to_modify} out of {target_samples} samples with label {target_label} "
                   f"({(num_to_modify/target_samples)*100:.1f}% of target samples, "
                   f"{(num_to_modify/total_samples)*100:.1f}% of total samples)")
    else:
        logger.info(f"Modified {num_to_modify} out of {total_samples} samples ({injection_percentage*100:.1f}%)")
    
    logger.info(f"Trigger tokens: {trigger_tokens}")
    logger.info(f"Injection position: {injection_position}")
    
    return modified_dataset 

def _filter_by_labels(dataset: Dataset, label_column: str, filter_labels: List[Union[int, str]]) -> Dataset:
    """Helper function to filter dataset by labels."""
    # Convert filter_labels to same type as in dataset
    label_type = type(dataset[0][label_column])
    filter_labels = [label_type(label) for label in filter_labels]
    
    # Create filter function
    def keep_example(example):
        return example[label_column] in filter_labels
    
    # Filter dataset
    filtered = dataset.filter(keep_example)
    
    # Log statistics
    total = len(dataset)
    kept = len(filtered)
    logger.info(f"Filtered dataset from {total} to {kept} examples "
                f"({(kept/total)*100:.1f}% kept)")
    
    return filtered 