"""
Base dataset loader class for custom dataset implementations.

This module contains the abstract base class that all custom dataset loaders
should inherit from.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
from torch.utils.data import Dataset
from datasets import DatasetDict
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class BaseDatasetLoader(ABC):
    """Abstract base class for all dataset loaders."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = None

    @abstractmethod
    def load(self, *args, **kwargs) -> Union[Dataset, DatasetDict]:
        """Load a dataset. Must be implemented by subclasses."""
        pass

    def tokenize(self, batch: Dict[str, List[Any]], text_field: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Tokenize a batch of data.

        Args:
            batch: A dictionary containing lists of data.
            text_field: The name of the field(s) containing text to tokenize.

        Returns:
            A dictionary containing the tokenized 'input_ids' and 'attention_mask'.
        """
        if isinstance(text_field, str):
            text_to_tokenize = batch[text_field]
            # Handle None values in single text field
            text_to_tokenize = [text if text is not None else "" for text in text_to_tokenize]
        else:
            # Handle multiple text fields (e.g., premise, hypothesis)
            # This simple concatenation works for many cases, but might need
            # to be customized in specific loaders.
            # Filter out None values and join the remaining text
            text_to_tokenize = []
            for example in zip(*(batch[field] for field in text_field)):
                # Filter out None values and join the remaining text
                valid_texts = [text for text in example if text is not None]
                if valid_texts:
                    text_to_tokenize.append(" ".join(valid_texts))
                else:
                    text_to_tokenize.append("")  # Empty string if all texts are None

        return self.tokenizer(
            text_to_tokenize,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ) 