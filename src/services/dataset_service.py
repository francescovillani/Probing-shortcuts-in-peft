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
import hashlib

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
    def load(self) -> Union[Dataset, DatasetDict]:
        """Load the dataset."""
        pass

    def tokenize(self, batch: Dict, text_fields: Union[str, List[str]] = None) -> Dict:
        """Tokenize the input batch based on text fields."""
        if isinstance(text_fields, str):
            text_fields = [text_fields]
        
        # If text_fields is provided, use it explicitly
        if text_fields:
            # Check if all specified fields exist in the batch
            missing_fields = [field for field in text_fields if field not in batch]
            if missing_fields:
                raise ValueError(f"Text fields not found in batch: {missing_fields}")
            
            # If multiple fields, pass them as separate arguments
            if len(text_fields) > 1:
                field_values = [batch[field] for field in text_fields]
                return self.tokenizer(
                    *field_values,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                )
            else:
                # Single field
                return self.tokenizer(
                    batch[text_fields[0]],
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                )

        # Fallback: GLUE-style paired inputs
        if "premise" in batch and "hypothesis" in batch:
            return self.tokenizer(
                batch["premise"],
                batch["hypothesis"],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
        # Fallback: Default to first string field found
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
        split: Optional[str] = None,
        trust_remote_code: bool = False
    ) -> Union[Dataset, DatasetDict]:
        """Load a dataset from HuggingFace Hub."""
        logger.info(f"Loading HuggingFace dataset: {dataset_name} (config: {config}, split: {split})")
        self.dataset = load_dataset(dataset_name, config, split=split, trust_remote_code=trust_remote_code)
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
    ) -> Union[Dataset, DatasetDict]:
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
        seed: int = 42,
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
        target_label: Union[int, str, List[Union[int, str]]],
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
    
    def apply_splitting(
        self,
        dataset: Dataset,
        train_size: float,
        test_size: Optional[float] = None,
        split_seed: int = 42,
        stratify_by: Optional[str] = None
    ) -> Tuple[Dataset, Dataset]:
        """
        Apply deterministic dataset splitting to create train and test splits.
        
        This method ensures that the same split is always produced for the same
        dataset and configuration, making it safe to use for both training and
        validation datasets that need to be split from the same source.
        
        Args:
            dataset: The dataset to split
            train_size: Proportion of data for training (0.1 to 0.9)
            test_size: Proportion of data for testing. If None, calculated as 1 - train_size
            split_seed: Random seed for reproducible splitting
            stratify_by: Column name to stratify by (e.g., 'label' for balanced splits)
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Calculate test_size if not provided
        if test_size is None:
            test_size = 1.0 - train_size
        
        # Validate proportions
        if train_size + test_size > 1.0:
            raise ValueError(f"train_size ({train_size}) + test_size ({test_size}) cannot exceed 1.0")
        
        # Handle DatasetDict objects by extracting the appropriate dataset
        if isinstance(dataset, DatasetDict):
            # If it's a DatasetDict, we need to extract the actual dataset
            # For HuggingFace datasets, we typically want the "train" split
            if "train" in dataset:
                actual_dataset = dataset["train"]
                logger.info(f"Extracted 'train' split from DatasetDict with {len(actual_dataset)} samples")
            else:
                # If no "train" split, take the first available split
                split_name = list(dataset.keys())[0]
                actual_dataset = dataset[split_name]
                logger.info(f"Extracted '{split_name}' split from DatasetDict with {len(actual_dataset)} samples")
        else:
            actual_dataset = dataset
        
        # Remove duplicates before splitting
        initial_size = len(actual_dataset)
        logger.info(f"Original dataset size before deduplication: {initial_size}")
        
        # Create a unique identifier for each sample based on all columns
        # This approach works well for text datasets where we want to identify truly duplicate samples
        def create_row_hash(example):
            # Convert all values to strings and normalize whitespace for consistent deduplication
            normalized_example = {}
            for key, value in example.items():
                if isinstance(value, str):
                    # Normalize whitespace: strip and collapse multiple spaces/newlines
                    normalized_value = ' '.join(str(value).split())
                else:
                    normalized_value = value
                normalized_example[key] = normalized_value
            
            row_str = str(sorted(normalized_example.items()))
            # Use SHA-256 for deterministic hashing across sessions
            return {"row_hash": hashlib.sha256(row_str.encode('utf-8')).hexdigest()}
        
        # Add hash column to identify duplicates
        hashed_dataset = actual_dataset.map(create_row_hash)
        
        # Get unique hashes and their indices
        unique_hashes = set()
        unique_indices = []
        
        for idx, example in enumerate(hashed_dataset):
            row_hash = example["row_hash"]
            if row_hash not in unique_hashes:
                unique_hashes.add(row_hash)
                unique_indices.append(idx)
        
        # Select only unique samples
        actual_dataset = actual_dataset.select(unique_indices)
        deduplicated_size = len(actual_dataset)
        duplicates_removed = initial_size - deduplicated_size
        
        logger.info(f"Removed {duplicates_removed} duplicate samples")
        logger.info(f"Dataset size after deduplication: {deduplicated_size}")
        
        total_size = len(actual_dataset)
        train_count = int(total_size * train_size)
        test_count = int(total_size * test_size)
        
        logger.info(f"Splitting deduplicated dataset of {total_size} samples: {train_count} train, {test_count} test")
        logger.info(f"Split proportions: train={train_size:.2f}, test={test_size:.2f}")
        logger.info(f"Using split seed: {split_seed}")
        
        # Create a deterministic split using the provided seed
        # We'll use the dataset's built-in train_test_split method for consistency
        split_datasets = actual_dataset.train_test_split(
            train_size=train_size,
            test_size=test_size,
            seed=split_seed,
            stratify_by_column=stratify_by
        )
        
        train_dataset = split_datasets["train"]
        test_dataset = split_datasets["test"]
        
        logger.info(f"Successfully split dataset: {len(train_dataset)} train samples, {len(test_dataset)} test samples")
        
        return train_dataset, test_dataset
    
    def create_split_datasets(
        self,
        base_config: DatasetConfig,
        train_config: Optional[DatasetConfig] = None,
        test_config: Optional[DatasetConfig] = None
    ) -> Tuple[Dataset, Dataset]:
        """
        Create coordinated train and test datasets from the same source dataset.
        
        This method ensures that both datasets are split from the same source using
        the same splitting configuration, guaranteeing consistency.
        
        Args:
            base_config: Base dataset configuration with splitting enabled
            train_config: Optional override config for training dataset (batch_size, poisoning, etc.)
            test_config: Optional override config for test dataset (batch_size, poisoning, etc.)
            
        Returns:
            Tuple of (train_dataset, test_dataset)
            
        Raises:
            ValueError: If base_config doesn't have splitting enabled
        """
        if not base_config.splitting or not base_config.splitting.enabled:
            raise ValueError("Base config must have splitting enabled")
        
        logger.info(f"Creating coordinated train/test splits from dataset: {base_config.name}")
        
        # Load the base dataset
        base_dataset = self._load_dataset(base_config.model_dump())
        
        # Apply splitting to get train and test datasets
        train_dataset, test_dataset = self.apply_splitting(
            dataset=base_dataset,
            train_size=base_config.splitting.train_size,
            test_size=base_config.splitting.test_size,
            split_seed=base_config.splitting.split_seed,
            stratify_by=base_config.splitting.stratify_by
        )
        
        # Apply any train-specific configurations
        if train_config:
            if train_config.poisoning and train_config.poisoning.enabled:
                logger.info("Applying poisoning to training dataset")
                poison_params = train_config.poisoning.model_dump(exclude={'enabled'})
                train_dataset = self.apply_poisoning(
                    dataset=train_dataset,
                    **poison_params
                )
        
        # Apply any test-specific configurations
        if test_config:
            if test_config.poisoning and test_config.poisoning.enabled:
                logger.info("Applying poisoning to test dataset")
                poison_params = test_config.poisoning.model_dump(exclude={'enabled'})
                test_dataset = self.apply_poisoning(
                    dataset=test_dataset,
                    **poison_params
                )
        
        logger.info(f"Successfully created coordinated splits: {len(train_dataset)} train, {len(test_dataset)} test")
        
        return train_dataset, test_dataset
    
    def save_dataset(self, dataset: Dataset, path: str) -> None:
        """Save dataset to disk."""
        logger.info(f"Saving dataset to {path}")
        dataset.save_to_disk(path)
    
    def extract_debug_samples(self, dataset: Dataset, dataset_name: str, num_samples: int = 5, text_field: Union[str, List[str]] = None) -> List[Dict[str, Any]]:
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
            text_content = self._extract_text_from_example(example, text_field)
            
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
    
    def _extract_text_from_example(self, example: Dict[str, Any], text_field: Union[str, List[str]]) -> Dict[str, str]:
        """
        Extract text content from a dataset example, handling various column structures.
        
        Args:
            example: A single example from the dataset
            text_field: Specific text field(s) to extract, or None for auto-detection
            
        Returns:
            Dictionary with text field names as keys and text content as values
        """
        text_fields = {}
        
        # If text_field is specified, use it directly
        if text_field is not None:
            if isinstance(text_field, str):
                text_field_list = [text_field]
            else:
                text_field_list = text_field
            
            # Extract specified text fields
            for field_name in text_field_list:
                if field_name in example:
                    text_fields[field_name] = str(example[field_name])
                else:
                    logger.warning(f"Specified text field '{field_name}' not found in example keys: {list(example.keys())}")
            
            # Return early if we found the specified fields
            if text_fields:
                return text_fields
        
        # Fallback: auto-detection logic
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
        text_field: Union[str, List[str]] = None,
        label_field: str = "label",
        max_train_size: Optional[int] = None,
        extract_debug_samples: bool = True,
        num_debug_samples: int = 5,
        return_raw_datasets: bool = False,
    ) -> Tuple[Optional[DataLoader], Dict[str, DataLoader], Dict[str, List[Dict[str, Any]]], Optional[Dataset]]:
        """Prepare train and validation dataloaders with debug samples, optionally returning raw datasets."""
        train_loader = None
        debug_samples = {}
        raw_train_dataset = None
        
        if train_config is not None:
            text_field = train_config.text_field
            label_field = train_config.label_field
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
            
            # Store raw dataset if requested (AFTER all transformations but BEFORE tokenization)
            if return_raw_datasets:
                raw_train_dataset = train_dataset
                logger.info(f"Stored raw training dataset with {len(raw_train_dataset)} samples")
            
            # Extract debug samples AFTER poisoning but BEFORE tokenization
            if extract_debug_samples:
                debug_samples["training"] = self.extract_debug_samples(train_dataset, "training", num_debug_samples, text_field)
            
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
                    debug_samples[val_name] = self.extract_debug_samples(val_dataset, val_name, num_debug_samples, text_field)
                
                val_dataset = self._process_dataset(val_dataset, text_field, label_field)
                val_loaders[val_name] = DataLoader(
                    val_dataset,
                    batch_size=val_config.batch_size,
                    shuffle=False
                )
                logger.info(f"Created validation dataloader for {val_name} with batch size {val_config.batch_size}")

        return train_loader, val_loaders, debug_samples, raw_train_dataset

    def _load_dataset(self, config: Dict) -> Dataset:
        """Helper to load a dataset from a config dictionary."""
        config_obj = DatasetConfig.model_validate(config)

        if config_obj.dataset_type and config_obj.dataset_type in self._custom_loaders:
            loader = self._get_custom_loader(config_obj.dataset_type)
            dataset = loader.load()
        elif config_obj.is_hf:
            dataset = self.hf_loader.load(
                dataset_name=config_obj.name,
                config=config_obj.config,
                split=config_obj.split,
                trust_remote_code=config_obj.trust_remote_code,
            )
        elif config_obj.is_local:
            dataset = self.local_loader.load(
                dataset_path=config_obj.name,
                split=config_obj.split,
            )
        else:
            raise ValueError(f"Could not determine how to load dataset: {config_obj.name}")
        
        # Handle dataset splitting if configured
        if config_obj.splitting and config_obj.splitting.enabled:
            logger.info(f"Applying dataset splitting for dataset: {config_obj.name}")
            
            base_dataset = dataset
            
            # Apply splitting to get train and test datasets
            train_dataset, test_dataset = self.apply_splitting(
                dataset=base_dataset,
                train_size=config_obj.splitting.train_size,
                test_size=config_obj.splitting.test_size,
                split_seed=config_obj.splitting.split_seed,
                stratify_by=config_obj.splitting.stratify_by
            )
            
            # Return the appropriate split based on the config
            if config_obj.splitting.split == "train":
                return train_dataset
            elif config_obj.splitting.split == "test":
                return test_dataset
            else:
                # If no specific split requested, return the train split by default
                logger.info("No specific split requested, returning train split by default")
                return train_dataset
        
        return dataset

    def _get_custom_loader(self, dataset_type: str) -> BaseDatasetLoader:
        """Get or create an instance of a custom loader."""
        if dataset_type not in self._custom_loader_instances:
            if dataset_type not in self._custom_loaders:
                raise ValueError(f"No loader registered for dataset type: {dataset_type}")
            loader_class = self._custom_loaders[dataset_type]
            self._custom_loader_instances[dataset_type] = loader_class(self.tokenizer, self.max_length)
        return self._custom_loader_instances[dataset_type]

    def _process_dataset(
        self, 
        dataset: Dataset, 
        text_field: Union[str, List[str]],
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
        target_label: Union[int, str, List[Union[int, str]]],
        injection_percentage: float = 0.1,
        injection_position: str = 'start',
        label_column: str = "label",
        filter_labels: Optional[List[Union[int, str]]] = None
    ) -> Union[Dataset, DatasetDict]:
        """Inject triggers into dataset (extracted from dataset_modifiers)."""
        
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
        target_label: Union[int, str, List[Union[int, str]]],
        label_column: str = "label"
    ) -> Dataset:
        """Helper function to inject triggers into a single dataset split."""
        
        # Create trigger text
        trigger_text = " ".join(trigger_tokens)
        
        # Calculate trigger token length for truncation-aware injection
        # We need to account for the trigger tokens plus potential separators
        trigger_tokenized = self.tokenizer.encode(trigger_text, add_special_tokens=False)
        trigger_token_length = len(trigger_tokenized)
        
        # Reserve space for special tokens (CLS, SEP, etc.) - typically 2-3 tokens
        special_tokens_reserve = 3
        
        logger.info(f"Trigger text: '{trigger_text}' ({trigger_token_length} tokens)")
        logger.info(f"Max sequence length: {self.max_length}, reserving {special_tokens_reserve} for special tokens")
        
        # Get indices of samples with target label(s)
        if label_column not in dataset.column_names:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        # Normalize target_label to a list for consistent processing
        if isinstance(target_label, (int, str)):
            target_labels = [target_label]
        else:
            target_labels = target_label
        
        # Convert target_labels to same type as in dataset
        dataset_label_type = type(dataset[0][label_column])
        target_labels_typed = [dataset_label_type(label) for label in target_labels]
        
        # Use a local random instance for deterministic sampling
        local_random = random.Random(self.seed)

        # Get indices of samples with any of the target labels
        candidate_indices = [
            i for i, example in enumerate(dataset) 
            if example[label_column] in target_labels_typed
        ]
        logger.info(f"Found {len(candidate_indices)} samples with target label(s) {target_labels}")

        if not candidate_indices:
            raise ValueError(f"No samples found with any of the target labels: {target_labels}")

        # Select from candidate indices
        num_to_modify = min(int(len(dataset) * injection_percentage), len(candidate_indices))
        indices_to_modify = local_random.sample(candidate_indices, num_to_modify)

        # Create a set for faster lookup
        indices_set = set(indices_to_modify)
        
        def modify_text(text: str, idx: int) -> str:
            """Helper to modify a single text string with truncation-aware injection."""
            if idx not in indices_set:
                return text

            if injection_position == 'start':
                return f"{trigger_text} {text}"
            elif injection_position == 'end':
                # For end injection, ensure the trigger fits within max_length
                # Calculate how much space we have for the original text
                available_tokens_for_text = self.max_length - trigger_token_length - special_tokens_reserve
                
                if available_tokens_for_text <= 0:
                    logger.warning(f"Trigger is too long ({trigger_token_length} tokens) for max_length {self.max_length}")
                    # Fallback: just append trigger and let tokenizer truncate
                    return f"{text} {trigger_text}"
                
                # Tokenize the original text to check its length
                text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
                
                if len(text_tokens) <= available_tokens_for_text:
                    # Original text fits, just append trigger
                    return f"{text} {trigger_text}"
                else:
                    # Need to truncate original text to make room for trigger
                    truncated_tokens = text_tokens[:available_tokens_for_text]
                    truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    
                    logger.debug(f"Truncated text from {len(text_tokens)} to {len(truncated_tokens)} tokens to fit trigger")
                    return f"{truncated_text} {trigger_text}"
            else:  # random
                # For random injection: first find where text would be truncated,
                # then inject trigger randomly within the safe space
                words = text.split()

                # Calculate how many tokens we have available for the original text
                available_tokens_for_text = self.max_length - trigger_token_length - special_tokens_reserve

                if available_tokens_for_text <= 0:
                    # Trigger takes up all available space
                    logger.warning(f"Trigger is too long for random injection with max_length {self.max_length}")
                    modified_text = trigger_text
                else:
                    # Find the safe truncation boundary for the original text
                    # Tokenize progressively to find where we hit the limit
                    safe_word_count = len(words)  # Start with all words

                    for i in range(len(words)):
                        partial_text = " ".join(words[:i+1])
                        partial_tokens = self.tokenizer.encode(partial_text, add_special_tokens=False)

                        if len(partial_tokens) > available_tokens_for_text:
                            safe_word_count = i  # Previous position was the last safe one
                            break

                    # If no words fit, use empty text
                    if safe_word_count == 0:
                        logger.warning(f"No space for original text with trigger injection")
                        modified_text = trigger_text
                    else:
                        # Now inject trigger at random position within the safe word range
                        safe_words = words[:safe_word_count]
                        insert_pos = local_random.randint(0, len(safe_words))

                        # Insert trigger at the random position
                        trigger_words = trigger_text.split()
                        final_words = safe_words[:insert_pos] + trigger_words + safe_words[insert_pos:]
                        modified_text = " ".join(final_words)

                    logger.debug(f"Random injection: inserted trigger at word position {insert_pos} out of {safe_word_count} safe words")

                return modified_text
        
        # Create modification function for all text columns
        def modify_example(example, idx):
            modified = example.copy()
            
            columns_found = [col for col in text_column_names if col in example]
            
            for column in columns_found:
                modified[column] = modify_text(example[column], idx)

            # If this example was supposed to be modified but no column was found, log a warning
            if not columns_found and idx in indices_set:
                logger.warning(
                    f"Trigger injection failed for sample index {idx}. "
                    f"None of the text_column_names {text_column_names} were found in "
                    f"the example keys: {list(example.keys())}"
                )
            
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
        logger.info(f"Modified {len(indices_to_modify)} out of {target_samples} samples with label(s) {target_labels} "
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

    def create_clean_triggered_dataloaders(
        self,
        config: DatasetConfig,
        text_field: Union[str, List[str]],
        label_field: str = "label"
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create matched clean and triggered dataloaders for the same examples.
        
        This method is specifically designed for cosine similarity analysis, where we need
        to compare the exact same examples with and without triggers.
        
        Args:
            config: Dataset configuration with poisoning settings
            text_field: Text field to use for tokenization
            label_field: Label field name
            
        Returns:
            Tuple of (clean_dataloader, triggered_dataloader) with examples in the same order
        """
        if not config.poisoning or not config.poisoning.enabled:
            raise ValueError("Poisoning configuration required for clean/triggered comparison")
        
        # Load the base dataset
        base_dataset = self.load_dataset(config)
        
        # Create clean version (no poisoning)
        clean_dataset = self._process_dataset(base_dataset, text_field, label_field)
        
        # Create triggered version by applying poisoning
        poison_params = config.poisoning.model_dump(exclude={'enabled'})
        triggered_dataset = self.apply_poisoning(
            dataset=base_dataset,
            **poison_params
        )
        
        # Filter to only include triggered examples (has_trigger == 1)
        # This ensures we only compare examples that actually have triggers
        def has_trigger_filter(example):
            return example.get("has_trigger", 0) == 1
        
        triggered_indices = []
        for i, example in enumerate(triggered_dataset):
            if has_trigger_filter(example):
                triggered_indices.append(i)
        
        logger.info(f"Found {len(triggered_indices)} triggered examples for similarity analysis")
        
        if len(triggered_indices) == 0:
            raise ValueError("No triggered examples found in dataset")
        
        # Select the same indices from both clean and triggered datasets
        clean_subset = clean_dataset.select(triggered_indices)
        triggered_subset = triggered_dataset.select(triggered_indices)
        
        # Process the triggered subset (apply tokenization)
        triggered_subset = self._process_dataset(triggered_subset, text_field, label_field)
        
        # Create dataloaders with same batch size and NO shuffling to maintain order
        clean_dataloader = DataLoader(
            clean_subset,
            batch_size=config.batch_size,
            shuffle=False  # Critical: must maintain order for pairwise comparison
        )
        
        triggered_dataloader = DataLoader(
            triggered_subset,
            batch_size=config.batch_size,
            shuffle=False  # Critical: must maintain order for pairwise comparison
        )
        
        logger.info(f"Created clean/triggered dataloaders with {len(clean_subset)} examples each")
        
        return clean_dataloader, triggered_dataloader 