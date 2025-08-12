"""
Custom dataset loader for LIAR dataset.

This loader handles the LIAR and LIAR-PLUS datasets with their 6-class truthfulness labels,
and provides options for binary classification by merging classes into true/false.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import PreTrainedTokenizer
import sys
import os
from collections import Counter

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from .base_loader import BaseDatasetLoader

logger = logging.getLogger(__name__)


class LiarDatasetLoader(BaseDatasetLoader):
    """
    Custom loader for LIAR and LIAR-PLUS datasets.
    
    This loader handles both the original LIAR dataset and the enhanced LIAR-PLUS dataset:
    - Original LIAR: 14 tab-separated columns
    - LIAR-PLUS: 16 tab-separated columns (adds id, json_id, and justification fields)
    
    Each TSV file contains truthfulness labels:
    false, mostly-false, half-true, mostly-true, true, pants-fire
    
    The loader supports both original 6-class and binary true/false classification.
    """
    
    # Original 6 classes in LIAR dataset
    ORIGINAL_LABELS = ['false', 'mostly-false', 'half-true', 'mostly-true', 'true', 'pants-fire']
    
    # LIAR-PLUS uses different labels
    LIAR_PLUS_LABELS = ['false', 'barely-true', 'half-true', 'mostly-true', 'true', 'pants-fire']
    
    # Binary classification mapping - works for both datasets
    BINARY_TRUE_LABELS = ['true', 'mostly-true', 'half-true']
    BINARY_FALSE_LABELS = ['false', 'mostly-false', 'barely-true', 'pants-fire']
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        super().__init__(tokenizer, max_length)
        self.available_text_fields = ['statement', 'subject', 'combined', 'justification']
        self.label_field = 'labels'
        
    def load(
        self, 
        dataset_path: str,
        text_field: Union[str, List[str]] = "statement",
        binary_classification: bool = True,
        balance_binary: bool = False,
        split: Optional[str] = None,
        seed: int = 42,
        combine_statement_subject: bool = False,
        statement_subject_separator: str = " [SEP] "
    ) -> Union[Dataset, DatasetDict]:
        """
        Load the LIAR or LIAR-PLUS dataset from TSV files.
        
        Args:
            dataset_path: Path to the LIAR dataset directory containing train.tsv, valid.tsv, test.tsv
            text_field: Text field(s) to use for classification. 
                       Available: ['statement', 'subject', 'combined', 'justification'] or list of fields
                       'combined' merges statement and subject with separator
            binary_classification: If True, convert to binary true/false classification
            balance_binary: If True and binary_classification=True, balance the classes to 50/50
            split: Optional specific split to return ('train', 'valid', 'test', or None for all)
            seed: Random seed for balancing operations
            combine_statement_subject: Whether to create a 'combined' field from statement + subject
            statement_subject_separator: Separator used when combining statement and subject
            
        Returns:
            Dataset or DatasetDict object ready for training/evaluation
        """
        # Construct full paths to TSV files
        if not os.path.isabs(dataset_path):
            # If relative path, assume it's relative to project root
            project_root = Path(__file__).parent.parent.parent
            dataset_dir = project_root / dataset_path
        else:
            dataset_dir = Path(dataset_path)
            
        train_path = dataset_dir / "train.tsv"
        valid_path = dataset_dir / "valid.tsv"
        test_path = dataset_dir / "test.tsv"
        
        # Check if files exist
        required_files = {"train.tsv": train_path, "valid.tsv": valid_path, "test.tsv": test_path}
        missing_files = [name for name, path in required_files.items() if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing LIAR dataset files: {missing_files}")
            
        logger.info(f"Loading LIAR dataset from: {dataset_dir}")
        
        # Load the dataframes
        datasets = {}
        try:
            for split_name, file_path in required_files.items():
                split_key = split_name.replace('.tsv', '')
                df = self._load_tsv_file(file_path)
                datasets[split_key] = df
                logger.info(f"Loaded {len(df)} examples from {split_name}")
                
        except Exception as e:
            raise ValueError(f"Failed to load TSV files from {dataset_dir}: {str(e)}")
        
        # Process each split
        processed_datasets = {}
        for split_name, df in datasets.items():
            processed_df = self._process_dataframe(
                df, 
                binary_classification=binary_classification,
                balance_binary=balance_binary,
                combine_statement_subject=combine_statement_subject,
                statement_subject_separator=statement_subject_separator,
                seed=seed,
                split_name=split_name
            )
            processed_datasets[split_name] = processed_df
        
        # Convert to HuggingFace datasets
        hf_datasets = {}
        for split_name, df in processed_datasets.items():
            dataset_dict = self._dataframe_to_dict(df, combine_statement_subject, statement_subject_separator)
            dataset = Dataset.from_dict(dataset_dict)
            
            # Set up proper labels
            if binary_classification:
                label_names = ["True", "False"]  # Order matters: 0=True, 1=False
            else:
                # Use appropriate label set based on dataset type
                if hasattr(self, 'dataset_type') and self.dataset_type == 'liar-plus':
                    label_names = self.LIAR_PLUS_LABELS
                else:
                    label_names = self.ORIGINAL_LABELS
            
            dataset = dataset.cast_column("labels", ClassLabel(names=label_names))
            hf_datasets[split_name] = dataset
            
            logger.info(f"Created {split_name} dataset with {len(dataset)} examples")
            
        # Log overall statistics
        self._log_dataset_statistics(hf_datasets, binary_classification)
        
        # Store metadata
        for split_name, dataset in hf_datasets.items():
            dataset._custom_loader_type = 'liar'
            dataset.binary_classification = binary_classification
            dataset.text_fields = text_field if isinstance(text_field, list) else [text_field]
            
            if binary_classification:
                dataset.label_mapping = {'True': 0, 'False': 1}
            else:
                if hasattr(self, 'dataset_type') and self.dataset_type == 'liar-plus':
                    dataset.label_mapping = {label: idx for idx, label in enumerate(self.LIAR_PLUS_LABELS)}
                else:
                    dataset.label_mapping = {label: idx for idx, label in enumerate(self.ORIGINAL_LABELS)}
        
        # Return single dataset if specific split requested, otherwise DatasetDict
        if split:
            if split not in hf_datasets:
                raise ValueError(f"Split '{split}' not found. Available splits: {list(hf_datasets.keys())}")
            self.dataset = hf_datasets[split]
            return hf_datasets[split]
        else:
            dataset_dict = DatasetDict(hf_datasets)
            self.dataset = dataset_dict
            return dataset_dict
    
    def _load_tsv_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single TSV file and parse it into a DataFrame with proper column names."""
        # First, detect if this is LIAR-PLUS (16 columns) or original LIAR (14 columns)
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            num_columns = len(first_line.split('\t'))
        
        if num_columns == 16:
            # LIAR-PLUS format
            column_names = [
                'id', 'json_id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
                'state_info', 'party_affiliation', 'barely_true_counts', 
                'false_counts', 'half_true_counts', 'mostly_true_counts', 
                'pants_on_fire_counts', 'context', 'justification'
            ]
            logger.info(f"Detected LIAR-PLUS format (16 columns) in {file_path}")
        elif num_columns == 14:
            # Original LIAR format
            column_names = [
                'id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
                'state_info', 'party_affiliation', 'barely_true_counts', 
                'false_counts', 'half_true_counts', 'mostly_true_counts', 
                'pants_on_fire_counts', 'context'
            ]
            logger.info(f"Detected original LIAR format (14 columns) in {file_path}")
        else:
            logger.warning(f"Unexpected number of columns ({num_columns}) in {file_path}. Expected 14 or 16.")
            # Fall back to LIAR-PLUS format and see what happens
            column_names = [
                'id', 'json_id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
                'state_info', 'party_affiliation', 'barely_true_counts', 
                'false_counts', 'half_true_counts', 'mostly_true_counts', 
                'pants_on_fire_counts', 'context', 'justification'
            ]
        
        df = pd.read_csv(file_path, sep='\t', names=column_names, header=None)
        
        # Detect the dataset type based on labels found
        unique_labels = df['label'].unique()
        has_barely_true = 'barely-true' in unique_labels
        has_mostly_false = 'mostly-false' in unique_labels
        
        if has_barely_true and not has_mostly_false:
            self.dataset_type = 'liar-plus'
            expected_labels = set(self.LIAR_PLUS_LABELS)
            logger.info(f"Detected LIAR-PLUS dataset based on labels in {file_path}")
        elif has_mostly_false and not has_barely_true:
            self.dataset_type = 'liar'
            expected_labels = set(self.ORIGINAL_LABELS)
            logger.info(f"Detected original LIAR dataset based on labels in {file_path}")
        else:
            # Default to LIAR-PLUS if we can't determine clearly
            self.dataset_type = 'liar-plus'
            expected_labels = set(self.LIAR_PLUS_LABELS)
            logger.warning(f"Could not clearly determine dataset type in {file_path}. Defaulting to LIAR-PLUS.")
        
        # Validate labels
        unexpected_labels = set(unique_labels) - expected_labels
        if unexpected_labels:
            logger.warning(f"Found unexpected labels in {file_path}: {unexpected_labels}")
        
        # Add justification column if it doesn't exist (for backward compatibility with original LIAR)
        if 'justification' not in df.columns:
            df['justification'] = ""
            logger.info(f"Added empty 'justification' column for compatibility")
            
        return df
    
    def _process_dataframe(
        self, 
        df: pd.DataFrame, 
        binary_classification: bool,
        balance_binary: bool,
        combine_statement_subject: bool,
        statement_subject_separator: str,
        seed: int,
        split_name: str
    ) -> pd.DataFrame:
        """Process a dataframe by converting labels and balancing if needed."""
        df = df.copy()
        
        # Clean text fields
        df['statement'] = df['statement'].fillna("").astype(str)
        df['subject'] = df['subject'].fillna("").astype(str)
        df['justification'] = df['justification'].fillna("").astype(str)
        
        df['statement'] = df['statement'].str.replace(r'\s+', ' ', regex=True).str.strip()
        df['subject'] = df['subject'].str.replace(r'\s+', ' ', regex=True).str.strip()
        df['justification'] = df['justification'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Clean labels - remove rows with NaN or invalid labels
        df = df.dropna(subset=['label'])
        df = df[df['label'].notna()]
        df = df[df['label'] != 'nan']
        
        if len(df) == 0:
            raise ValueError(f"No valid examples found in {split_name} after cleaning labels")
        
        # Convert labels
        if binary_classification:
            df['labels'] = df['label'].apply(self._convert_to_binary_label)
            
            # Log original distribution
            original_dist = df['labels'].value_counts()
            logger.info(f"{split_name} original binary distribution - True: {original_dist.get(0, 0)} ({original_dist.get(0, 0)/len(df)*100:.1f}%), "
                       f"False: {original_dist.get(1, 0)} ({original_dist.get(1, 0)/len(df)*100:.1f}%)")
            
            # Balance if requested
            if balance_binary:
                df = self._balance_binary_classes(df, seed, split_name)
        else:
            # Create numeric labels for original 6-class
            if hasattr(self, 'dataset_type') and self.dataset_type == 'liar-plus':
                label_to_num = {label: idx for idx, label in enumerate(self.LIAR_PLUS_LABELS)}
                label_names = self.LIAR_PLUS_LABELS
            else:
                label_to_num = {label: idx for idx, label in enumerate(self.ORIGINAL_LABELS)}
                label_names = self.ORIGINAL_LABELS
                
            df['labels'] = df['label'].map(label_to_num)
            
            # Remove any rows where label mapping failed (returned NaN)
            before_count = len(df)
            df = df.dropna(subset=['labels'])
            after_count = len(df)
            if before_count != after_count:
                logger.warning(f"Removed {before_count - after_count} rows with unmappable labels in {split_name}")
            
            # Convert labels to integers
            df['labels'] = df['labels'].astype(int)
            
            # Log distribution
            label_dist = df['labels'].value_counts().sort_index()
            dist_str = ", ".join([f"{label_names[idx]}: {count} ({count/len(df)*100:.1f}%)" 
                                for idx, count in label_dist.items() if idx < len(label_names)])
            logger.info(f"{split_name} label distribution - {dist_str}")
        
        return df
    
    def _convert_to_binary_label(self, original_label: str) -> int:
        """Convert original 6-class label to binary (0=True, 1=False)."""
        if original_label in self.BINARY_TRUE_LABELS:
            return 0  # True
        elif original_label in self.BINARY_FALSE_LABELS:
            return 1  # False
        else:
            logger.warning(f"Unknown label '{original_label}', treating as False")
            return 1
    
    def _balance_binary_classes(self, df: pd.DataFrame, seed: int, split_name: str) -> pd.DataFrame:
        """Balance binary classes to achieve 50/50 distribution."""
        np.random.seed(seed)
        
        true_examples = df[df['labels'] == 0]
        false_examples = df[df['labels'] == 1]
        
        min_count = min(len(true_examples), len(false_examples))
        
        # Sample equal numbers from each class
        balanced_true = true_examples.sample(n=min_count, random_state=seed)
        balanced_false = false_examples.sample(n=min_count, random_state=seed)
        
        balanced_df = pd.concat([balanced_true, balanced_false], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle
        
        logger.info(f"{split_name} balanced binary distribution - True: {min_count} (50.0%), False: {min_count} (50.0%)")
        logger.info(f"{split_name} dataset size after balancing: {len(balanced_df)} (reduced from {len(df)})")
        
        return balanced_df
    
    def _dataframe_to_dict(
        self, 
        df: pd.DataFrame, 
        combine_statement_subject: bool, 
        statement_subject_separator: str
    ) -> Dict[str, List]:
        """Convert processed dataframe to dictionary format for HuggingFace Dataset."""
        dataset_dict = {
            'statement': df['statement'].tolist(),
            'subject': df['subject'].tolist(),
            'labels': df['labels'].tolist(),
            'speaker': df['speaker'].fillna("").astype(str).tolist(),
            'context': df['context'].fillna("").astype(str).tolist(),
            'party_affiliation': df['party_affiliation'].fillna("").astype(str).tolist(),
            'justification': df['justification'].fillna("").astype(str).tolist(),
        }
        
        # Add json_id if it exists (LIAR-PLUS format)
        if 'json_id' in df.columns:
            dataset_dict['json_id'] = df['json_id'].fillna("").astype(str).tolist()
        
        # Create combined field if requested
        if combine_statement_subject:
            combined_text = (df['statement'] + statement_subject_separator + df['subject']).str.strip()
            dataset_dict['combined'] = combined_text.tolist()
            logger.info(f"Created 'combined' text field using separator: '{statement_subject_separator}'")
        
        return dataset_dict
    
    def _log_dataset_statistics(self, datasets: Dict[str, Dataset], binary_classification: bool):
        """Log overall statistics for the loaded datasets."""
        total_examples = sum(len(ds) for ds in datasets.values())
        logger.info(f"Total examples across all splits: {total_examples}")
        
        split_sizes = {name: len(ds) for name, ds in datasets.items()}
        logger.info(f"Split sizes: {split_sizes}")
        
        if binary_classification:
            logger.info("Binary classification mode enabled")
        else:
            logger.info("Original 6-class classification mode")
    
    def get_text_fields(self) -> List[str]:
        """Return available text fields for this dataset."""
        return self.available_text_fields
        
    def get_label_mapping(self, binary: bool = False) -> Dict[str, int]:
        """Return the label mapping used by this dataset."""
        if binary:
            return {'True': 0, 'False': 1}
        else:
            if hasattr(self, 'dataset_type') and self.dataset_type == 'liar-plus':
                return {label: idx for idx, label in enumerate(self.LIAR_PLUS_LABELS)}
            else:
                return {label: idx for idx, label in enumerate(self.ORIGINAL_LABELS)}
        
    def tokenize(self, batch: Dict, text_fields: Union[str, List[str]] = None) -> Dict:
        """
        Tokenize the input batch for LIAR dataset.
        
        Overrides the base tokenize method to handle LIAR-specific text fields.
        """
        if text_fields is None:
            # Default to statement if available, otherwise try other fields
            if 'statement' in batch:
                text_fields = 'statement'
            elif 'combined' in batch:
                text_fields = 'combined'
            elif 'subject' in batch:
                text_fields = 'subject'
            elif 'justification' in batch:
                text_fields = 'justification'
            else:
                raise ValueError(f"No suitable text field found in batch. Available fields: {list(batch.keys())}")
        
        # Use parent class tokenization with the determined text fields
        return super().tokenize(batch, text_fields)
    
    def get_dataset_stats(self) -> Dict[str, any]:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
            
        if isinstance(self.dataset, DatasetDict):
            # Multiple splits
            stats = {
                "splits": list(self.dataset.keys()),
                "total_samples": sum(len(ds) for ds in self.dataset.values()),
                "split_sizes": {name: len(ds) for name, ds in self.dataset.items()},
                "features": list(next(iter(self.dataset.values())).features.keys()),
                "text_fields": self.available_text_fields
            }
            
            # Add label distribution for each split
            for split_name, split_dataset in self.dataset.items():
                label_counts = Counter(split_dataset['labels'])
                stats[f"{split_name}_label_distribution"] = dict(label_counts)
                
        else:
            # Single dataset
            stats = {
                "total_samples": len(self.dataset),
                "label_distribution": dict(Counter(self.dataset['labels'])),
                "features": list(self.dataset.features.keys()),
                "text_fields": self.available_text_fields
            }
            
            # Calculate text length statistics for available text fields
            for field in ['statement', 'justification']:
                if field in self.dataset.column_names:
                    field_lengths = [len(text) for text in self.dataset[field]]
                    if field_lengths:  # Only add stats if there are non-empty texts
                        stats[f'{field}_text_stats'] = {
                            "mean_length": sum(field_lengths) / len(field_lengths),
                            "max_length": max(field_lengths),
                            "min_length": min(field_lengths)
                        }
        
        return stats 