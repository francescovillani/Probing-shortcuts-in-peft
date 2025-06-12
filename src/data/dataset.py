from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
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

class LocalLoader(BaseDatasetLoader):
    """Loader for local datasets."""
    
    def load(self, path: str, split: Optional[str] = None) -> DatasetDict:
        """Load a dataset from local storage."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
            
        logger.info(f"Loading local dataset from: {path} (split: {split})")
        self.dataset = load_from_disk(path)
        if split and split in self.dataset:
            self.dataset = self.dataset[split]
        return self.dataset

class DatasetManager:
    """Main class for dataset operations."""
    
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
        self.local_loader = LocalLoader(tokenizer, max_length)

    def prepare_dataset(
        self,
        train_config: Dict,
        val_config: Optional[Dict] = None,
        text_field: Optional[str] = None,
        label_field: str = "label",
        max_train_size: Optional[int] = None,
    ) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        """
        Prepare train and validation dataloaders.
        
        Args:
            train_config: Dict containing train dataset configuration
                {
                    "name": str,  # dataset name or path
                    "config": Optional[str],  # dataset config name
                    "batch_size": int,  # batch size for training
                    "is_local": bool,  # whether dataset is local
                    "split": Optional[str]  # dataset split to use
                }
            val_config: Dict of validation dataset configurations
            text_field: Field containing the text to tokenize
            label_field: Field containing the labels
            max_train_size: Maximum number of training examples
        """
        # Load training dataset
        train_dataset = self._load_dataset(train_config)
        
        # Handle training set size limitation
        if max_train_size:
            train_dataset = train_dataset.shuffle(seed=self.seed)
            train_dataset = train_dataset.select(range(min(max_train_size, len(train_dataset))))
            logger.info(f"Limited training dataset to {len(train_dataset)} examples")

        # Tokenize and format training data
        train_dataset = self._process_dataset(train_dataset, text_field, label_field)
        
        # Create training dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.get("batch_size", 16),
            shuffle=True
        )
        logger.info(f"Created training dataloader with batch size {train_config.get('batch_size', 16)}")

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

    def _load_dataset(self, config: Dict) -> Dataset:
        """Load dataset based on configuration."""
        if config.get("is_local", False):
            return self.local_loader.load(
                config["name"],
                config.get("split")
            )
        return self.hf_loader.load(
            config["name"],
            config.get("config"),
            config.get("split")
        )

    def _process_dataset(
        self, 
        dataset: Dataset, 
        text_field: Optional[str],
        label_field: str
    ) -> Dataset:
        """Process dataset with tokenization and formatting."""
        # Tokenize
        dataset = dataset.map(
            lambda batch: self.hf_loader.tokenize(batch, text_field),
            batched=True
        )
        
        # Rename label field if needed
        if label_field in dataset.column_names and label_field != "labels":
            dataset = dataset.rename_column(label_field, "labels")
            
        # Set format for PyTorch
        dataset.set_format(
            "torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        return dataset 