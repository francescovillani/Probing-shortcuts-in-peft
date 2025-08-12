"""
Dataset-specific prompt templates for decoder models.

This module contains prompt templates for various datasets and utilities
for applying them to transform raw dataset examples into prompted text.
"""

import re
from itertools import tee
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datasets import Dataset

logger = logging.getLogger(__name__)


class DatasetPromptTemplates:
    """Registry of dataset-specific prompt templates."""
    
    # MNLI (Natural Language Inference)
    MNLI_TEMPLATE = """Determine the relationship between premise and hypothesis.
    Premise: {premise}
    Hypothesis: {hypothesis}
    Choose one of the following relationships: entailment, contradiction, neutral
    Answer:"""

    # SST-2 (Sentiment Analysis)
    SST2_TEMPLATE = "Is the sentiment of this sentence positive or negative?\nSentence: {sentence}\nSentiment:"

    # Hatespeech (Hate Speech Detection)
    HATESPEECH_TEMPLATE = "Classify if the text is hate speech, offensive, or neither.\nText: {tweet}\nLabel:"
    
    # Generic templates for common patterns
    SINGLE_TEXT_CLASSIFICATION_TEMPLATE = """Classify the following text.
    Text: {text}
    Answer:"""

    DUAL_TEXT_CLASSIFICATION_TEMPLATE = """Analyze the relationship between the following two texts.
    Text 1: {text1}
    Text 2: {text2}
    Answer:"""

    @classmethod
    def get_dataset_templates(cls) -> Dict[str, str]:
        """Get mapping of dataset names to their templates."""
        return {
            # GLUE tasks
            "mnli": cls.MNLI_TEMPLATE,
            "sst2": cls.SST2_TEMPLATE,
            "hate_speech": cls.HATESPEECH_TEMPLATE,
            # Common patterns
            "single_text": cls.SINGLE_TEXT_CLASSIFICATION_TEMPLATE,
            "dual_text": cls.DUAL_TEXT_CLASSIFICATION_TEMPLATE,
        }

    @classmethod
    def get_label_mappings(cls) -> Dict[str, Dict[int, str]]:
        """Get label mappings for datasets with known label structures."""
        return {
            "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
            "sst2": {0: "negative", 1: "positive"},
            "hatespeech": {0: "hate_speech", 1: "offensive_language", 2: "neither"},
        }


class PromptTemplateService:
    """Service for applying dataset-specific prompting to datasets."""
    
    def __init__(self, tokenizer=None, max_length=512):
        self.templates = DatasetPromptTemplates.get_dataset_templates()
        self.label_mappings = DatasetPromptTemplates.get_label_mappings()
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def _calculate_template_overhead(self, template: str, answer_prefix: str) -> int:
        """Calculate the token overhead from template structure (excluding content placeholders)."""
        if not self.tokenizer:
            # Rough estimate if no tokenizer available
            return len(template.split()) + len(answer_prefix.split())
        
        # Create a minimal template with single character placeholders
        placeholder_pattern = r'\{([^}]+)\}'
        minimal_template = re.sub(placeholder_pattern, 'x', template)
        
        # Add answer prefix
        minimal_prompted = minimal_template + f"\n\n{answer_prefix}"
        
        # Tokenize to get actual overhead
        overhead_tokens = self.tokenizer.encode(minimal_prompted, add_special_tokens=False)
        return len(overhead_tokens)
    
    def _truncate_content_intelligently(
        self, 
        format_dict: Dict[str, str], 
        template: str, 
        answer_prefix: str,
        max_length: int
    ) -> Dict[str, str]:
        """
        Intelligently truncate content to fit within max_length while preserving template structure.
        
        For multi-field templates (like MNLI), this prioritizes fields and balances truncation.
        """
        if not self.tokenizer:
            return format_dict  # Can't truncate without tokenizer
        
        # Calculate template overhead (structure + answer prefix)
        template_overhead = self._calculate_template_overhead(template, answer_prefix)
        
        # Reserve space for special tokens (CLS, SEP, etc.)
        special_tokens_reserve = 3
        
        # Available space for actual content
        available_content_tokens = max_length - template_overhead - special_tokens_reserve
        
        if available_content_tokens <= 0:
            logger.warning(f"Template overhead ({template_overhead}) + special tokens ({special_tokens_reserve}) exceeds max_length ({max_length})")
            # Return minimal content
            return {k: "..." for k in format_dict.keys() if isinstance(format_dict[k], str)}
        
        # Identify text content fields (ignore non-string values)
        text_fields = {k: v for k, v in format_dict.items() if isinstance(v, str) and v.strip()}
        
        if not text_fields:
            return format_dict
        
        # Calculate current content token usage
        total_content_tokens = 0
        field_token_counts = {}
        
        for field, content in text_fields.items():
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            field_token_counts[field] = len(tokens)
            total_content_tokens += len(tokens)
        
        # If content already fits, return as-is
        if total_content_tokens <= available_content_tokens:
            logger.debug(f"Content fits within limits: {total_content_tokens} <= {available_content_tokens} tokens")
            return format_dict
        
        logger.debug(f"Content truncation needed: {total_content_tokens} tokens > {available_content_tokens} available")
        logger.debug(f"Field token counts: {field_token_counts}")
        
        # Apply intelligent truncation based on field types and template structure
        truncated_fields = self._apply_field_truncation_strategy(
            text_fields, 
            field_token_counts, 
            available_content_tokens,
            template
        )
        
        # Log truncation results
        truncated_token_counts = {}
        for field, content in truncated_fields.items():
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            truncated_token_counts[field] = len(tokens)
        
        logger.debug(f"After truncation - field token counts: {truncated_token_counts}")
        logger.debug(f"Total tokens after truncation: {sum(truncated_token_counts.values())}")
        
        # Update format_dict with truncated content
        result_dict = format_dict.copy()
        for field, truncated_content in truncated_fields.items():
            result_dict[field] = truncated_content
        
        return result_dict
    
    def _apply_field_truncation_strategy(
        self,
        text_fields: Dict[str, str],
        field_token_counts: Dict[str, int],
        available_tokens: int,
        template: str
    ) -> Dict[str, str]:
        """
        Apply field-specific truncation strategy based on template type and field importance.
        """
        # Define field priority for different template types
        field_priorities = self._get_field_priorities(text_fields, template)
        
        # If only one field, truncate it to fit
        if len(text_fields) == 1:
            field_name = list(text_fields.keys())[0]
            content = text_fields[field_name]
            return {field_name: self._truncate_single_field(content, available_tokens)}
        
        # For multiple fields, use balanced truncation with priorities
        return self._truncate_multiple_fields(text_fields, field_token_counts, available_tokens, field_priorities)
    
    def _get_field_priorities(self, text_fields: Dict[str, str], template: str) -> Dict[str, int]:
        """
        Assign field truncation priorities based on field length.
        Lower number = higher priority (truncate less).
        Shorter fields get higher priority.
        """
        # Compute lengths for each field
        field_lengths = {field: len(content) for field, content in text_fields.items()}
        # Sort fields by length (shortest first)
        sorted_fields = sorted(field_lengths.items(), key=lambda x: x[1])
        # Assign priorities: shortest gets 0, next shortest 1, etc.
        priorities = {}
        for idx, (field, _) in enumerate(sorted_fields):
            priorities[field] = idx
        return priorities
    
    def _truncate_single_field(self, content: str, max_tokens: int) -> str:
        """Truncate a single field to fit within token limit."""
        if not self.tokenizer:
            # Rough word-based truncation
            words = content.split()
            estimated_tokens = len(words) * 1.3  # Rough estimate
            if estimated_tokens <= max_tokens:
                return content
            target_words = int(max_tokens / 1.3)
            return " ".join(words[:target_words])
        
        tokens = self.tokenizer.encode(content, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return content
        
        # Truncate tokens and decode back
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    def _truncate_multiple_fields(
        self,
        text_fields: Dict[str, str],
        field_token_counts: Dict[str, int],
        available_tokens: int,
        field_priorities: Dict[str, int]
    ) -> Dict[str, str]:
        """
        Truncate multiple fields using priority-based allocation.
        """
        # Sort fields by priority (lower number = higher priority)
        sorted_fields = sorted(text_fields.keys(), key=lambda x: field_priorities.get(x, 1))
        
        # Allocate tokens starting with highest priority fields
        allocations = {}
        remaining_tokens = available_tokens
        
        # First pass: give minimum reasonable allocation to all fields
        min_tokens_per_field = 10  # Minimum tokens to keep any field meaningful
        
        for field in sorted_fields:
            field_tokens = field_token_counts[field]
            # Allocate either the minimum or what the field needs (if less than minimum)
            allocation = min(min_tokens_per_field, field_tokens, remaining_tokens)
            allocations[field] = allocation
            remaining_tokens -= allocation
            
            if remaining_tokens <= 0:
                break
        
        # Second pass: distribute remaining tokens by priority
        for field in sorted_fields:
            if remaining_tokens <= 0:
                break
                
            field_tokens = field_token_counts[field]
            current_allocation = allocations.get(field, 0)
            
            # How many more tokens could this field use?
            additional_needed = field_tokens - current_allocation
            
            if additional_needed > 0:
                # Give additional tokens up to what's needed or what's available
                additional_allocation = min(additional_needed, remaining_tokens)
                allocations[field] = current_allocation + additional_allocation
                remaining_tokens -= additional_allocation
        
        # Apply truncation based on allocations
        truncated_fields = {}
        for field, content in text_fields.items():
            target_tokens = allocations.get(field, 0)
            truncated_fields[field] = self._truncate_single_field(content, target_tokens)
        
        logger.debug(f"Token allocations: {allocations}")
        
        return truncated_fields
    
    def detect_dataset_type(self, dataset: Dataset, dataset_name: str, text_fields: Union[str, List[str]]) -> Optional[str]:
        """
        Auto-detect the appropriate template for a dataset.
        
        Args:
            dataset: The dataset to analyze
            dataset_name: Name of the dataset
            text_fields: Text field(s) in the dataset
            
        Returns:
            Template key or None if no suitable template found
        """
        # Check if dataset name contains known task identifiers
        dataset_name_lower = dataset_name.lower()
        
        # Direct matches for GLUE tasks
        glue_tasks = ["mnli", "sst2", "hate_speech"]
        for task in glue_tasks:
            if task in dataset_name_lower:
                logger.info(f"Detected dataset type '{task}' from name: {dataset_name}")
                return task
        
        # Check based on field structure
        if isinstance(text_fields, list):
            if len(text_fields) == 2:
                
                text_fields_lower = [f.lower() for f in text_fields]
                
                # Check for specific MNLI pattern
                if "premise" in text_fields_lower and "hypothesis" in text_fields_lower:
                    logger.info("Detected MNLI-style premise-hypothesis structure")
                    return "mnli"
                
                # Generic dual text pattern
                logger.info("Detected dual text structure")
                return "dual_text"
                
        elif isinstance(text_fields, str):
            # Single text field - could be sentiment, classification, etc.
            text_field_lower = text_fields.lower()
            
            if "sentence" in text_field_lower:
                # Could be SST-2 or similar sentiment task
                if "sst" in dataset_name_lower or "sentiment" in dataset_name_lower:
                    logger.info("Detected sentiment analysis structure")
                    return "sst2"
                    
            logger.info("Detected single text classification structure")
            return "single_text"
        
        logger.warning(f"Could not auto-detect template for dataset: {dataset_name}")
        return None
    
    def apply_prompting(
        self,
        dataset: Dataset,
        dataset_name: str,
        text_fields: Union[str, List[str]],
        template: Optional[str] = None,
        answer_prefix: str = "Answer:",
    ) -> Dataset:
        """
        Apply prompting to a dataset for decoder models.
        
        Args:
            dataset: The dataset to apply prompting to
            dataset_name: Name of the dataset for template detection
            text_fields: Text field(s) to use in prompting
            template: Custom template (if None, will auto-detect)
            answer_prefix: Prefix before the answer
            
        Returns:
            Dataset with 'prompted_text' field added
        """
        # Determine template to use
        if template is None:
            template_key = self.detect_dataset_type(dataset, dataset_name, text_fields)
            if template_key is None:
                logger.warning(f"No template found for dataset {dataset_name}, skipping prompting")
                return dataset
            template = self.templates[template_key]
        
        logger.info(f"Applying prompting to dataset {dataset_name}")
        logger.info(f"Template preview: {template[:100]}...")
        
        def create_prompted_text(example: Dict[str, Any]) -> Dict[str, Any]:
            """Create prompted text for a single example."""
            try:
                # Extract text values for template formatting
                format_dict = {}
                
                if isinstance(text_fields, str):
                    # Single text field
                    format_dict[text_fields] = example.get(text_fields, "")
                    
                    # Only add aliases if they're actually used in the template and not already present
                    content = example.get(text_fields, "")
                    template_placeholders = re.findall(r'\{(\w+)\}', template)
                    
                    for alias in ["text", "sentence"]:
                        if (alias in template_placeholders and 
                            alias not in format_dict and 
                            alias != text_fields):
                            format_dict[alias] = content
                else:
                    # Multiple text fields
                    for field in text_fields:
                        format_dict[field] = example.get(field, "")
                
                # Add other fields that might be in the template
                for key, value in example.items():
                    if isinstance(value, str) and key not in format_dict:
                        format_dict[key] = value
                
                # Apply intelligent truncation if tokenizer is available
                if self.tokenizer and self.max_length:
                    format_dict = self._truncate_content_intelligently(
                        format_dict, template, answer_prefix, self.max_length
                    )
                
                # Format the template
                try:
                    prompted_text = template.format(**format_dict)
                except KeyError as e:
                    logger.warning(f"Template formatting failed for field {e}, using fallback")
                    # Fallback: just use available fields
                    available_fields = {k: v for k, v in format_dict.items() if isinstance(v, str)}
                    prompted_text = template.format(**available_fields)
                
                # Ensure the answer prefix is present
                if not prompted_text.rstrip().endswith(answer_prefix.rstrip()):
                    prompted_text = prompted_text.rstrip() + f"\n\n{answer_prefix}"
                
                example["prompted_text"] = prompted_text
                
            except Exception as e:
                logger.error(f"Error creating prompted text: {e}")
                # Fallback to original text
                if isinstance(text_fields, str):
                    example["prompted_text"] = f"{example.get(text_fields, '')}\n\n{answer_prefix}"
                else:
                    text_parts = [example.get(field, "") for field in text_fields]
                    example["prompted_text"] = f"{' '.join(text_parts)}\n\n{answer_prefix}"
            
            return example
        
        # Apply prompting to the dataset
        prompted_dataset = dataset.map(create_prompted_text, desc="Applying prompts")
        
        logger.info(f"Successfully applied prompting to {len(prompted_dataset)} examples")
        
        return prompted_dataset