from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Type
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer
import torch
import os
import logging

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
        self.dataset = load_dataset(dataset_name, config, split=split)
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

class DatasetManager:
    """Main class for dataset operations."""
    
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

    def prepare_dataset(
        self,
        train_config: Optional[Dict] = None,
        val_config: Optional[Dict] = None,
        text_field: Optional[str] = None,
        label_field: str = "label",
        max_train_size: Optional[int] = None,
    ) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        """Prepare train and validation dataloaders."""
        # Load training dataset
        train_loader = None
        if train_config is not None:
            train_dataset = self._load_dataset(train_config)
            # trigger_config = train_dataset.trigger_config
            # Handle training set size limitation, reintroduce the config as subset removes it
            if max_train_size:
                train_dataset = train_dataset.shuffle(seed=self.seed)
                train_dataset = train_dataset.select(range(min(max_train_size, len(train_dataset))))
                logger.info(f"Limited training dataset to {len(train_dataset)} examples")
                # train_dataset.trigger_config = trigger_config

            # Tokenize and format training data
            train_dataset = self._process_dataset(train_dataset, text_field, label_field)
            
            # Create training dataloader
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_config.get("batch_size", 16),
                shuffle=True
            )
            logger.info(f"Created training dataloader with batch size {train_config.get('batch_size', 16)}")
        logger.info(f"No training dataset provided")

        # Handle validation datasets
        val_loaders = {}
        if val_config:
            for val_name, val_cfg in val_config.items():
                logger.info(f"Processing validation dataset: {val_name}")
                val_dataset = self._load_dataset(val_cfg)
                val_dataset = self._process_dataset(val_dataset, text_field, label_field)
                val_loaders[val_name] = DataLoader(
                    val_dataset,
                    batch_size=val_cfg.get("batch_size", 32),
                    shuffle=False
                )
                logger.info(f"Created validation dataloader for {val_name} with batch size {val_cfg.get('batch_size', 32)}")

        return train_loader, val_loaders

    def _get_custom_loader(self, dataset_type: str) -> BaseDatasetLoader:
        """Get or create a custom dataset loader instance."""
        if dataset_type not in self._custom_loader_instances:
            if dataset_type not in self._custom_loaders:
                raise ValueError(f"No loader registered for dataset type: {dataset_type}")
            loader_class = self._custom_loaders[dataset_type]
            self._custom_loader_instances[dataset_type] = loader_class(self.tokenizer, self.max_length)
        return self._custom_loader_instances[dataset_type]

    def _load_dataset(self, config: Dict) -> Dataset:
        """Load dataset based on configuration."""
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

