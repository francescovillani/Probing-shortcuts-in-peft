"""
Base dataset loader class for custom dataset implementations.

This module contains the abstract base class that all custom dataset loaders
should inherit from.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset
from datasets import DatasetDict
from transformers import PreTrainedTokenizer

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