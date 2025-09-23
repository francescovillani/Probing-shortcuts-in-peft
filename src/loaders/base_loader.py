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
        Tokenize a batch of data with intelligent handling of sentence pairs.

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
            
            return self.tokenizer(
                text_to_tokenize,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
        else:
            # Handle multiple text fields with intelligent sentence pair detection
            if len(text_field) == 2 and self._is_sentence_pair_scenario(text_field):
                # Use tokenizer's sentence pair functionality for better handling
                field1, field2 = text_field
                
                # Handle None values
                text1_list = [text if text is not None else "" for text in batch[field1]]
                text2_list = [text if text is not None else "" for text in batch[field2]]
                
                return self.tokenizer(
                    text1_list,
                    text2_list,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                )
            else:
                # Fall back to concatenation for other multi-field scenarios
                # This maintains backward compatibility for cases that don't fit sentence pairs
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
    
    def _is_sentence_pair_scenario(self, text_fields: List[str]) -> bool:
        """
        Determine if the given text fields represent a sentence pair scenario.
        
        Args:
            text_fields: List of field names to check.
            
        Returns:
            True if this looks like a sentence pair scenario, False otherwise.
        """
        if len(text_fields) != 2:
            return False
            
        field_pairs = [
            # Common sentence pair field combinations
            {"premise", "hypothesis"},
            {"sentence1", "sentence2"},
            {"text_a", "text_b"},
            {"question", "context"},
            {"query", "passage"},
            {"sent1", "sent2"},
            {"question1", "question2"},
            {"context1", "context2"},
            {"passage1", "passage2"},
            {"title1", "title2"},
            {"text1", "text2"},
            {"statement1", "statement2"},
            {"subject1", "subject2"},
            {"date1", "date2"},
            {"justification1", "justification2"},
            {"id1", "id2"},
            {"json_id1", "json_id2"},
        ]
        
        field_set = set(text_fields)
        return any(field_set == pair for pair in field_pairs) 