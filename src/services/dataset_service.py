"""
Dataset service for unified dataset operations.

This service handles dataset loading, poisoning, tokenization, and dataloader creation.
It consolidates functionality from the original DatasetManager and dataset_modifiers.
"""

import random
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Type, Any
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer
from pathlib import Path
import os
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import DatasetConfig

logger = logging.getLogger(__name__)


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loading operations."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = None

    @abstractmethod
    def load(self) -> DatasetDict:
        """Load the dataset."""
        pass

    def tokenize(self, batch: Dict, text_fields: Union[str, List[str]] = None) -> Dict:
        """Tokenize the input batch based on text fields."""
        if isinstance(text_fields, str):
            text_fields = [text_fields]
        
        # Handle GLUE-style paired inputs
        if "premise" in batch and "hypothesis" in batch:
            return self.tokenizer(
                batch["premise"],
                batch["hypothesis"],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )

        # Handle single text field
        if text_fields:
            for field in text_fields:
                if field in batch:
                    return self.tokenizer(
                        batch[field],
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=True,
                    )

        # Default to first string field found
        for key, value in batch.items():
            if isinstance(value, (str, list)) and value:
                return self.tokenizer(
                    value,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                )
        
        raise ValueError("No suitable text field found for tokenization.")


class HuggingFaceLoader(BaseDatasetLoader):
    """Loader for HuggingFace datasets."""
    
    def load(
        self, 
        dataset_name: str, 
        config: Optional[str] = None, 
        split: Optional[str] = None
    ) -> DatasetDict:
        """Load a dataset from HuggingFace Hub."""
        logger.info(f"Loading HuggingFace dataset: {dataset_name} (config: {config}, split: {split})")
        self.dataset = load_dataset(dataset_name, config, split=split, trust_remote_code=True)
        return self.dataset


class LocalDatasetLoader(BaseDatasetLoader):
    """Loader for locally saved datasets."""
    
    def _load_trigger_config(self, dataset_path: str) -> Optional[Dict]:
        """Load trigger configuration if it exists."""
        trigger_config_path = os.path.join(dataset_path, "trigger_config.txt")
        if not os.path.exists(trigger_config_path):
            return None
            
        config = {}
        with open(trigger_config_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    value = value.strip()
                    # Handle list values
                    if value.startswith('[') and value.endswith(']'):
                        value = [v.strip().strip("'") for v in value[1:-1].split(',')]
                    # Handle None values
                    elif value == 'None':
                        value = None
                    # Handle numeric values
                    elif value.replace('.', '').isdigit():
                        value = float(value) if '.' in value else int(value)
                    config[key.strip()] = value
        return config
    
    def load(
        self,
        dataset_path: str,
        split: Optional[str] = None
    ) -> DatasetDict:
        """Load a dataset from local disk."""
        logger.info(f"Loading local dataset from: {dataset_path}")
        try:
            dataset = load_from_disk(dataset_path)
            if split is not None:
                dataset = dataset[split] if isinstance(dataset, DatasetDict) else dataset
            
            # Load trigger config if available
            trigger_config = self._load_trigger_config(dataset_path)
            if trigger_config:
                logger.info(f"Found trigger configuration: {trigger_config}")
                # Attach trigger config to the dataset object
                dataset.trigger_config = trigger_config
            
            self.dataset = dataset
            return self.dataset
        except Exception as e:
            raise ValueError(f"Failed to load local dataset from {dataset_path}: {str(e)}")


class DatasetService:
    """
    Unified dataset service providing:
    - Dataset loading from various sources
    - Dataset poisoning/modification
    - Tokenization and preprocessing
    - DataLoader creation
    """
    
    # Registry of custom dataset loaders
    _custom_loaders: Dict[str, Type[BaseDatasetLoader]] = {}
    
    @classmethod
    def register_loader(cls, dataset_type: str, loader_class: Type[BaseDatasetLoader]):
        """Register a custom dataset loader."""
        cls._custom_loaders[dataset_type] = loader_class
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.hf_loader = HuggingFaceLoader(tokenizer, max_length)
        self.local_loader = LocalDatasetLoader(tokenizer, max_length)
        self._custom_loader_instances = {}

    def load_dataset(self, config: DatasetConfig) -> Dataset:
        """Load dataset based on configuration."""
        return self._load_dataset(config.model_dump())
    
    def apply_poisoning(
        self,
        dataset: Dataset,
        text_column_names: List[str],
        trigger_tokens: List[str],
        target_label: Union[int, str],
        injection_percentage: float = 0.1,
        injection_position: str = 'start',
        label_column: str = "label",
        filter_labels: Optional[List[Union[int, str]]] = None
    ) -> Dataset:
        """Apply poisoning/trigger injection to a dataset."""
        return self._inject_trigger_into_dataset(
            dataset=dataset,
            text_column_names=text_column_names,
            trigger_tokens=trigger_tokens,
            target_label=target_label,
            injection_percentage=injection_percentage,
            injection_position=injection_position,
            label_column=label_column,
            filter_labels=filter_labels
        )
    
    def save_dataset(self, dataset: Dataset, path: str) -> None:
        """Save dataset to disk."""
        logger.info(f"Saving dataset to {path}")
        dataset.save_to_disk(path)
    
    def extract_debug_samples(self, dataset: Dataset, dataset_name: str, num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Extract a deterministic set of sample examples from a dataset for debugging purposes.
        Prioritizes samples with triggers (poisoned samples) when available.
        
        Args:
            dataset: The dataset to sample from
            dataset_name: Name of the dataset for logging
            num_samples: Number of samples to extract (default: 5)
            
        Returns:
            List of dictionaries containing raw text and metadata
        """
        if len(dataset) == 0:
            logger.warning(f"Dataset {dataset_name} is empty, no debug samples extracted")
            return []
        
        # Use deterministic sampling based on seed and dataset name
        # This ensures the same samples are extracted across runs with the same seed
        import hashlib
        deterministic_seed = int(hashlib.md5(f"{self.seed}_{dataset_name}".encode()).hexdigest()[:8], 16)
        local_random = random.Random(deterministic_seed)
        
        total_samples = len(dataset)
        actual_num_samples = min(num_samples, total_samples)
        
        # Check if dataset has trigger information (from poisoning)
        has_trigger_column = "has_trigger" in dataset.column_names
        
        if has_trigger_column:
            # Separate indices by trigger status
            triggered_indices = []
            non_triggered_indices = []
            
            for i in range(total_samples):
                example = dataset[i]
                if example.get("has_trigger", 0):
                    triggered_indices.append(i)
                else:
                    non_triggered_indices.append(i)
            
            logger.info(f"Dataset '{dataset_name}' has {len(triggered_indices)} triggered samples and {len(non_triggered_indices)} non-triggered samples")
            
            # Prioritize triggered samples
            selected_indices = []
            
            # First, select from triggered samples (if any exist)
            if triggered_indices:
                num_triggered_to_select = min(actual_num_samples, len(triggered_indices))
                selected_triggered = local_random.sample(triggered_indices, num_triggered_to_select)
                selected_indices.extend(selected_triggered)
                logger.info(f"Selected {num_triggered_to_select} triggered samples for debug")
            
            # Then, fill remaining slots with non-triggered samples if needed
            remaining_slots = actual_num_samples - len(selected_indices)
            if remaining_slots > 0 and non_triggered_indices:
                num_non_triggered_to_select = min(remaining_slots, len(non_triggered_indices))
                selected_non_triggered = local_random.sample(non_triggered_indices, num_non_triggered_to_select)
                selected_indices.extend(selected_non_triggered)
                logger.info(f"Selected {num_non_triggered_to_select} non-triggered samples to fill remaining debug slots")
            
            indices = selected_indices
        else:
            # No trigger information available, use original random sampling
            indices = local_random.sample(range(total_samples), actual_num_samples)
            logger.info(f"No trigger information found in dataset '{dataset_name}', using random sampling")
        
        indices.sort()  # Sort for consistent ordering
        
        debug_samples = []
        for idx in indices:
            example = dataset[idx]
            
            # Extract text content from various possible column structures
            text_content = self._extract_text_from_example(example)
            
            # Extract label
            label = example.get("labels", example.get("label", "unknown"))
            
            # Check if this sample has a trigger (if poisoning was applied)
            has_trigger = example.get("has_trigger", 0)
            
            debug_sample = {
                "index": idx,
                "text_content": text_content,
                "label": label,
                "has_trigger": bool(has_trigger),
                "raw_example_keys": list(example.keys())
            }
            
            debug_samples.append(debug_sample)
            
        # Log summary of what was extracted
        triggered_count = sum(1 for sample in debug_samples if sample["has_trigger"])
        non_triggered_count = len(debug_samples) - triggered_count
        logger.info(f"Extracted {len(debug_samples)} debug samples from dataset '{dataset_name}': {triggered_count} triggered, {non_triggered_count} non-triggered")
        
        return debug_samples
    
    def _extract_text_from_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract text content from a dataset example, handling various column structures.
        
        Args:
            example: A single example from the dataset
            
        Returns:
            Dictionary with text field names as keys and text content as values
        """
        text_fields = {}
        
        # Common text field names to check
        text_columns = ["text", "sentence", "premise", "hypothesis", "sentence1", "sentence2", 
                       "question", "context", "passage", "document"]
        
        for col_name, value in example.items():
            # Skip non-text fields
            if col_name in ["labels", "label", "input_ids", "attention_mask", "token_type_ids", "has_trigger"]:
                continue
                
            # Include if it's a known text column or if it's a string value
            if col_name in text_columns or isinstance(value, str):
                text_fields[col_name] = str(value)
        
        # If no text fields found, try to find any string values
        if not text_fields:
            for col_name, value in example.items():
                if isinstance(value, str) and len(value) > 0:
                    text_fields[col_name] = value
        
        return text_fields

    def get_dataloaders(
        self,
        datasets: Dict[str, Dataset],
        batch_sizes: Dict[str, int],
        shuffle: Dict[str, bool] = None
    ) -> Dict[str, DataLoader]:
        """Create DataLoaders for multiple datasets."""
        if shuffle is None:
            shuffle = {name: True if 'train' in name else False for name in datasets.keys()}
        
        dataloaders = {}
        for name, dataset in datasets.items():
            dataloaders[name] = DataLoader(
                dataset,
                batch_size=batch_sizes.get(name, 32),
                shuffle=shuffle.get(name, False)
            )
            logger.info(f"Created dataloader for {name} with batch size {batch_sizes.get(name, 32)}")
        
        return dataloaders

    def prepare_datasets(
        self,
        train_config: Optional[DatasetConfig] = None,
        val_configs: Optional[Dict[str, DatasetConfig]] = None,
        text_field: Optional[str] = None,
        label_field: str = "label",
        max_train_size: Optional[int] = None,
        extract_debug_samples: bool = True,
        num_debug_samples: int = 5,
    ) -> Tuple[Optional[DataLoader], Dict[str, DataLoader], Dict[str, List[Dict[str, Any]]]]:
        """Prepare train and validation dataloaders with debug samples."""
        train_loader = None
        debug_samples = {}
        
        if train_config is not None:
            train_dataset = self.load_dataset(train_config)

            # Handle training set size limitation BEFORE poisoning
            if max_train_size:
                train_dataset = train_dataset.shuffle(seed=self.seed)
                train_dataset = train_dataset.select(range(min(max_train_size, len(train_dataset))))
                logger.info(f"Limited training dataset to {len(train_dataset)} examples BEFORE poisoning")

            # Apply poisoning if configured
            if train_config.poisoning and train_config.poisoning.enabled:
                logger.info("Applying poisoning to training dataset")
                poison_params = train_config.poisoning.model_dump(exclude={'enabled'})
                train_dataset = self.apply_poisoning(
                    dataset=train_dataset,
                    **poison_params
                )
            
            # Extract debug samples AFTER poisoning but BEFORE tokenization
            if extract_debug_samples:
                debug_samples["training"] = self.extract_debug_samples(train_dataset, "training", num_debug_samples)
            
            # Tokenize and format training data
            train_dataset = self._process_dataset(train_dataset, text_field, label_field)
            
            # Create training dataloader
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_config.batch_size,
                shuffle=True
            )
            logger.info(f"Created training dataloader with batch size {train_config.batch_size}")

        # Handle validation datasets
        val_loaders = {}
        if val_configs:
            for val_name, val_config in val_configs.items():
                logger.info(f"Processing validation dataset: {val_name}")
                val_dataset = self.load_dataset(val_config)

                # Apply poisoning if configured
                if val_config.poisoning and val_config.poisoning.enabled:
                    logger.info(f"Applying poisoning to validation dataset: {val_name}")
                    poison_params = val_config.poisoning.model_dump(exclude={'enabled'})
                    val_dataset = self.apply_poisoning(
                        dataset=val_dataset,
                        **poison_params
                    )
                
                # Extract debug samples AFTER poisoning but BEFORE tokenization
                if extract_debug_samples:
                    debug_samples[val_name] = self.extract_debug_samples(val_dataset, val_name, num_debug_samples)
                
                val_dataset = self._process_dataset(val_dataset, text_field, label_field)
                val_loaders[val_name] = DataLoader(
                    val_dataset,
                    batch_size=val_config.batch_size,
                    shuffle=False
                )
                logger.info(f"Created validation dataloader for {val_name} with batch size {val_config.batch_size}")

        return train_loader, val_loaders, debug_samples

    def _load_dataset(self, config: Dict) -> Dataset:
        """Load dataset based on configuration dictionary."""
        # Count how many loading methods are specified
        loading_methods = sum([
            config.get("is_local", False),
            config.get("is_hf", False),
            config.get("dataset_type") is not None
        ])
        
        if loading_methods == 0:
            # Default to HuggingFace if no method specified
            logger.info("No loading method specified, defaulting to HuggingFace")
            return self.hf_loader.load(
                config["name"],
                config.get("config"),
                config.get("split")
            )
        elif loading_methods > 1:
            raise ValueError(
                "Ambiguous dataset loading configuration. "
                "Please specify exactly one of: is_local=True, is_hf=True, or dataset_type"
            )
        
        # Now we know exactly one method is specified
        if config.get("is_local", False):
            # Load from local disk using load_from_disk
            return self.local_loader.load(
                config["name"],
                config.get("split")
            )
        elif config.get("is_hf", False):
            # Load from HuggingFace
            return self.hf_loader.load(
                config["name"],
                config.get("config"),
                config.get("split")
            )
        else:  # Must be dataset_type since we validated above
            # Load using custom loader
            loader = self._get_custom_loader(config["dataset_type"])
            dataset = loader.load(
                config["name"],
                config.get("split")
            )
            dataset._custom_loader_type = config["dataset_type"]
            return dataset

    def _get_custom_loader(self, dataset_type: str) -> BaseDatasetLoader:
        """Get or create a custom dataset loader instance."""
        if dataset_type not in self._custom_loader_instances:
            if dataset_type not in self._custom_loaders:
                raise ValueError(f"No loader registered for dataset type: {dataset_type}")
            loader_class = self._custom_loaders[dataset_type]
            self._custom_loader_instances[dataset_type] = loader_class(self.tokenizer, self.max_length)
        return self._custom_loader_instances[dataset_type]

    def _process_dataset(
        self, 
        dataset: Dataset, 
        text_field: Optional[str],
        label_field: str
    ) -> Dataset:
        """Process dataset with tokenization and formatting."""
        # Get the appropriate loader for tokenization
        if isinstance(dataset, DatasetDict):
            dataset = next(iter(dataset.values()))
        
        # Preserve trigger config if it exists
        trigger_config = getattr(dataset, 'trigger_config', None)
        
        # Determine which loader to use for tokenization
        if hasattr(dataset, '_custom_loader_type'):
            loader = self._get_custom_loader(dataset._custom_loader_type)
        else:
            loader = self.hf_loader
        
        # Tokenize using the appropriate loader
        dataset = dataset.map(
            lambda batch: loader.tokenize(batch, text_field),
            batched=True
        )
        
        # Rename label field if needed and if 'labels' doesn't already exist
        if label_field in dataset.column_names and label_field != "labels" and "labels" not in dataset.column_names:
            dataset = dataset.rename_column(label_field, "labels")
        
        # Restore trigger config
        if trigger_config is not None:
            dataset.trigger_config = trigger_config
            
        # Convert string labels to numeric values for MNLI
        if "labels" in dataset.column_names and isinstance(dataset["labels"][0], str):
            label_map = {
                "entailment": 0,
                "neutral": 1,
                "contradiction": 2
            }
            dataset = dataset.map(
                lambda example: {"labels": label_map[example["labels"]]},
                remove_columns=["labels"]
            )
            
        # Set format for PyTorch
        dataset.set_format(
            "torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        return dataset

    def _inject_trigger_into_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        text_column_names: List[str],
        trigger_tokens: List[str],
        target_label: Union[int, str],
        injection_percentage: float = 0.1,
        injection_position: str = 'start',
        label_column: str = "label",
        filter_labels: Optional[List[Union[int, str]]] = None
    ) -> Union[Dataset, DatasetDict]:
        """Inject triggers into dataset (extracted from dataset_modifiers)."""
        random.seed(self.seed)
        
        # Validate inputs
        if not 0 <= injection_percentage <= 1:
            raise ValueError("injection_percentage must be between 0 and 1")
        
        if injection_position not in ['start', 'end', 'random']:
            raise ValueError("injection_position must be one of: 'start', 'end', 'random'")
        
        # Handle both Dataset and DatasetDict
        if isinstance(dataset, DatasetDict):
            modified_dataset = DatasetDict({
                split: self._inject_trigger_into_split(
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
            modified_dataset = self._inject_trigger_into_split(
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
                    split: self._filter_by_labels(ds, label_column, filter_labels)
                    for split, ds in modified_dataset.items()
                })
            else:
                filtered_dataset = self._filter_by_labels(modified_dataset, label_column, filter_labels)
            
            modified_dataset = filtered_dataset
        
        return modified_dataset

    def _inject_trigger_into_split(
        self,
        dataset: Dataset,
        text_column_names: List[str],
        trigger_tokens: List[str],
        injection_percentage: float,
        injection_position: str,
        target_label: Union[int, str],
        label_column: str = "label"
    ) -> Dataset:
        """Helper function to inject triggers into a single dataset split."""
        
        # Create trigger text
        trigger_text = " ".join(trigger_tokens)
        
        # Get indices of samples with target label
        if label_column not in dataset.column_names:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        # Convert target_label to same type as in dataset
        target_label_typed = type(dataset[0][label_column])(target_label)
        
        # Get indices of samples with target label
        candidate_indices = [
            i for i, example in enumerate(dataset) 
            if example[label_column] == target_label_typed
        ]
        logger.info(f"Found {len(candidate_indices)} samples with target label {target_label}")
        
        if not candidate_indices:
            raise ValueError(f"No samples found with label {target_label}")
        
        # Select from candidate indices
        num_to_modify = int(len(candidate_indices) * injection_percentage)
        indices_to_modify = random.sample(candidate_indices, num_to_modify)
        
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
        target_samples = len(candidate_indices)
        logger.info(f"Modified {len(indices_to_modify)} out of {target_samples} samples with label {target_label} "
                   f"({(len(indices_to_modify)/target_samples)*100:.1f}% of target samples, "
                   f"{(len(indices_to_modify)/total_samples)*100:.1f}% of total samples)")
        
        logger.info(f"Trigger tokens: {trigger_tokens}")
        logger.info(f"Injection position: {injection_position}")
        
        return modified_dataset

    def _filter_by_labels(self, dataset: Dataset, label_column: str, filter_labels: List[Union[int, str]]) -> Dataset:
        """Helper function to filter dataset by labels."""
        # Convert filter_labels to same type as in dataset
        label_type = type(dataset[0][label_column])
        filter_labels = [label_type(label) for label in filter_labels]
        
        # Create filter function
        def keep_example(example):
            return example[label_column] in filter_labels
        
        # Filter dataset
        filtered = dataset.filter(keep_example)
        return filtered 