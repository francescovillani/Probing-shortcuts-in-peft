"""
Custom dataset loader for PAWS (Paraphrase Adversaries from Word Scrambling) dataset.

This loader handles the PAWS dataset which is designed for paraphrase identification.
The dataset contains pairs of sentences with binary labels indicating whether
they are paraphrases or not.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import PreTrainedTokenizer
import sys
import os
from collections import Counter

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from .base_loader import BaseDatasetLoader

logger = logging.getLogger(__name__)


class PawsDatasetLoader(BaseDatasetLoader):
    """
    Custom loader for PAWS (Paraphrase Adversaries from Word Scrambling) dataset.
    
    This loader handles the PAWS dataset which consists of:
    - train.tsv: Training examples with sentence pairs and labels
    - dev_and_test.tsv: Development and test examples combined
    
    Each TSV file contains columns: id, sentence1, sentence2, label
    The labels are binary: 0 = not paraphrases, 1 = paraphrases
    
    The loader supports different text field combinations and can split
    the dev_and_test file into separate validation and test sets.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        super().__init__(tokenizer, max_length)
        self.available_text_fields = ['sentence1', 'sentence2', 'combined']
        self.label_field = 'labels'
        
    def load(
        self, 
        dataset_path: str,
        text_field: Union[str, List[str]] = ["sentence1", "sentence2"],
        split: Optional[str] = None,
        dev_test_split_ratio: float = 1.0,
        combine_sentences: bool = True,
        sentence_separator: str = " [SEP] ",
        seed: int = 42,
        balance_classes: bool = False
    ) -> Union[Dataset, DatasetDict]:
        """
        Load the PAWS dataset from TSV files.
        
        Args:
            dataset_path: Path to the PAWS dataset directory containing train.tsv and dev_and_test.tsv
            text_field: Text field(s) to use for classification.
                       Available: ['sentence1', 'sentence2', 'combined', 'both_sentences'] or list of fields
                       'combined' merges sentence1 and sentence2 with separator
                       'both_sentences' returns both sentences as separate fields for pair classification
            split: Optional specific split to return ('train', 'validation', 'test', or None for all)
            dev_test_split_ratio: Ratio to split dev_and_test.tsv (0.5 = equal validation/test)
            combine_sentences: Whether to create a 'combined' field from sentence1 + sentence2
            sentence_separator: Separator used when combining sentences
            seed: Random seed for splitting and balancing operations
            balance_classes: If True, balance the classes to 50/50 distribution
            
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
        dev_test_path = dataset_dir / "dev_and_test.tsv"
        
        # Check if files exist
        required_files = {"train.tsv": train_path, "dev_and_test.tsv": dev_test_path}
        missing_files = [name for name, path in required_files.items() if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing PAWS dataset files: {missing_files}")
            
        logger.info(f"Loading PAWS dataset from: {dataset_dir}")
        
        # Load the dataframes
        datasets = {}
        try:
            # Load training data
            train_df = self._load_tsv_file(train_path)
            datasets['train'] = train_df
            logger.info(f"Loaded {len(train_df)} examples from train.tsv")
            
            # Load and split dev_and_test data
            dev_test_df = self._load_tsv_file(dev_test_path)
            logger.info(f"Loaded {len(dev_test_df)} examples from dev_and_test.tsv")
            
            # Split dev_and_test into validation and test
            dev_test_df = dev_test_df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle
            split_idx = int(len(dev_test_df) * dev_test_split_ratio)
            
            datasets['validation'] = dev_test_df[:split_idx].reset_index(drop=True)
            datasets['test'] = dev_test_df[split_idx:].reset_index(drop=True)
            
            logger.info(f"Split dev_and_test into validation: {len(datasets['validation'])} and test: {len(datasets['test'])}")
                
        except Exception as e:
            raise ValueError(f"Failed to load TSV files from {dataset_dir}: {str(e)}")
        
        # Process each split
        processed_datasets = {}
        for split_name, df in datasets.items():
            processed_df = self._process_dataframe(
                df, 
                balance_classes=balance_classes,
                combine_sentences=combine_sentences,
                sentence_separator=sentence_separator,
                seed=seed,
                split_name=split_name
            )
            processed_datasets[split_name] = processed_df
        
        # Convert to HuggingFace datasets
        hf_datasets = {}
        for split_name, df in processed_datasets.items():
            dataset_dict = self._dataframe_to_dict(df, combine_sentences, sentence_separator)
            dataset = Dataset.from_dict(dataset_dict)
            
            # Set up proper labels
            label_names = ["Not Paraphrase", "Paraphrase"]  # Order matters: 0=Not Paraphrase, 1=Paraphrase
            dataset = dataset.cast_column("labels", ClassLabel(names=label_names))
            hf_datasets[split_name] = dataset
            
            logger.info(f"Created {split_name} dataset with {len(dataset)} examples")
            
        # Log overall statistics
        self._log_dataset_statistics(hf_datasets)
        
        # Store metadata
        for split_name, dataset in hf_datasets.items():
            dataset._custom_loader_type = 'paws'
            dataset.text_fields = text_field if isinstance(text_field, list) else [text_field]
            dataset.label_mapping = {'Not Paraphrase': 0, 'Paraphrase': 1}
        
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
        # PAWS TSV format: id, sentence1, sentence2, label
        column_names = ['id', 'sentence1', 'sentence2', 'label']
        
        df = pd.read_csv(file_path, sep='\t', names=column_names, header=0)  # header=0 to skip first row
        
        # Convert label column to integer
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        
        # Clean the byte string prefixes if they exist (b'text' -> text or b"text" -> text)
        for col in ['sentence1', 'sentence2']:
            if col in df.columns:
                # Convert to string first
                df[col] = df[col].astype(str)
                
                # Log a sample to debug the format
                if len(df) > 0:
                    sample_value = str(df[col].iloc[0])
                    logger.debug(f"Sample {col} value before cleaning: {repr(sample_value)}")
                
                # Handle byte string representations that appear as literal strings
                # Pattern: b'text' -> text
                df[col] = df[col].str.replace(r"^b'(.+)'$", r"\1", regex=True)
                # Pattern: b"text" -> text  
                df[col] = df[col].str.replace(r'^b"(.+)"$', r"\1", regex=True)
                
                # Additional fallback: if string still starts with b', remove it
                df[col] = df[col].str.replace(r"^b'", "", regex=True)
                df[col] = df[col].str.replace(r"'$", "", regex=True)
                df[col] = df[col].str.replace(r'^b"', "", regex=True)
                df[col] = df[col].str.replace(r'"$', "", regex=True)
                
                # Log a sample after cleaning
                if len(df) > 0:
                    sample_value = str(df[col].iloc[0])
                    logger.debug(f"Sample {col} value after cleaning: {repr(sample_value)}")
        
        # Validate labels
        unique_labels = df['label'].unique()
        expected_labels = {0, 1}
        unexpected_labels = set(unique_labels) - expected_labels
        if unexpected_labels:
            logger.warning(f"Found unexpected labels in {file_path}: {unexpected_labels}")
        
        logger.info(f"Loaded PAWS data with {len(df)} examples from {file_path}")
        return df
    
    def _process_dataframe(
        self, 
        df: pd.DataFrame, 
        balance_classes: bool,
        combine_sentences: bool,
        sentence_separator: str,
        seed: int,
        split_name: str
    ) -> pd.DataFrame:
        """Process a dataframe by cleaning text and balancing if needed."""
        df = df.copy()
        
        # Clean text fields
        df['sentence1'] = df['sentence1'].fillna("").astype(str)
        df['sentence2'] = df['sentence2'].fillna("").astype(str)
        
        df['sentence1'] = df['sentence1'].str.replace(r'\s+', ' ', regex=True).str.strip()
        df['sentence2'] = df['sentence2'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Clean labels - remove rows with NaN or invalid labels
        df = df.dropna(subset=['label'])
        df = df[df['label'].notna()]
        df = df[df['label'].isin([0, 1])]  # Only keep valid binary labels
        
        if len(df) == 0:
            raise ValueError(f"No valid examples found in {split_name} after cleaning")
        
        # Convert labels to integers
        df['labels'] = df['label'].astype(int)
        
        # Log original distribution
        original_dist = df['labels'].value_counts()
        logger.info(f"{split_name} original distribution - Not Paraphrase: {original_dist.get(0, 0)} ({original_dist.get(0, 0)/len(df)*100:.1f}%), "
                   f"Paraphrase: {original_dist.get(1, 0)} ({original_dist.get(1, 0)/len(df)*100:.1f}%)")
        
        # Balance if requested
        if balance_classes:
            df = self._balance_classes(df, seed, split_name)
        
        return df
    
    def _balance_classes(self, df: pd.DataFrame, seed: int, split_name: str) -> pd.DataFrame:
        """Balance classes to achieve 50/50 distribution."""
        import numpy as np
        np.random.seed(seed)
        
        not_paraphrase_examples = df[df['labels'] == 0]
        paraphrase_examples = df[df['labels'] == 1]
        
        min_count = min(len(not_paraphrase_examples), len(paraphrase_examples))
        
        # Sample equal numbers from each class
        balanced_not_paraphrase = not_paraphrase_examples.sample(n=min_count, random_state=seed)
        balanced_paraphrase = paraphrase_examples.sample(n=min_count, random_state=seed)
        
        balanced_df = pd.concat([balanced_not_paraphrase, balanced_paraphrase], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle
        
        logger.info(f"{split_name} balanced distribution - Not Paraphrase: {min_count} (50.0%), Paraphrase: {min_count} (50.0%)")
        logger.info(f"{split_name} dataset size after balancing: {len(balanced_df)} (reduced from {len(df)})")
        
        return balanced_df
    
    def _dataframe_to_dict(
        self, 
        df: pd.DataFrame, 
        combine_sentences: bool, 
        sentence_separator: str
    ) -> Dict[str, List]:
        """Convert processed dataframe to dictionary format for HuggingFace Dataset."""
        dataset_dict = {
            'sentence1': df['sentence1'].tolist(),
            'sentence2': df['sentence2'].tolist(),
            'labels': df['labels'].tolist(),
            'id': df['id'].fillna("").astype(str).tolist(),
        }
        
        # Create combined field if requested
        if combine_sentences:
            combined_text = (df['sentence1'] + sentence_separator + df['sentence2']).str.strip()
            dataset_dict['combined'] = combined_text.tolist()
            logger.info(f"Created 'combined' text field using separator: '{sentence_separator}'")
        
        return dataset_dict
    
    def _log_dataset_statistics(self, datasets: Dict[str, Dataset]):
        """Log overall statistics for the loaded datasets."""
        total_examples = sum(len(ds) for ds in datasets.values())
        logger.info(f"Total examples across all splits: {total_examples}")
        
        split_sizes = {name: len(ds) for name, ds in datasets.items()}
        logger.info(f"Split sizes: {split_sizes}")
        
        # Log label distribution across all splits
        for split_name, dataset in datasets.items():
            label_counts = Counter(dataset['labels'])
            not_para_count = label_counts.get(0, 0)
            para_count = label_counts.get(1, 0)
            total = len(dataset)
            logger.info(f"{split_name} - Not Paraphrase: {not_para_count} ({not_para_count/total*100:.1f}%), "
                       f"Paraphrase: {para_count} ({para_count/total*100:.1f}%)")
    
    def get_text_fields(self) -> List[str]:
        """Return available text fields for this dataset."""
        return self.available_text_fields
        
    def get_label_mapping(self) -> Dict[str, int]:
        """Return the label mapping used by this dataset."""
        return {'Not Paraphrase': 0, 'Paraphrase': 1}
        
    def tokenize(self, batch: Dict, text_fields: Union[str, List[str]] = None) -> Dict:
        """
        Tokenize the input batch for PAWS dataset.
        
        For PAWS sentence pair classification, the recommended approach is to use
        text_fields=["sentence1", "sentence2"] which will be handled by the base
        class tokenize method for multi-field text.
        """
        if text_fields is None:
            # Default to sentence pair for PAWS
            text_fields = ["sentence1", "sentence2"]
        
        # Use parent class tokenization which handles multi-field text properly
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
                
            # Calculate text length statistics for sentence fields
            for split_name, split_dataset in self.dataset.items():
                for field in ['sentence1', 'sentence2']:
                    if field in split_dataset.column_names:
                        field_lengths = [len(text) for text in split_dataset[field] if text]
                        if field_lengths:
                            stats[f'{split_name}_{field}_text_stats'] = {
                                "mean_length": sum(field_lengths) / len(field_lengths),
                                "max_length": max(field_lengths),
                                "min_length": min(field_lengths)
                            }
                
        else:
            # Single dataset
            stats = {
                "total_samples": len(self.dataset),
                "label_distribution": dict(Counter(self.dataset['labels'])),
                "features": list(self.dataset.features.keys()),
                "text_fields": self.available_text_fields,
                "splits": None  # Single dataset has no splits
            }
            
            # Calculate text length statistics for sentence fields
            for field in ['sentence1', 'sentence2']:
                if field in self.dataset.column_names:
                    field_lengths = [len(text) for text in self.dataset[field] if text]
                    if field_lengths:
                        stats[f'{field}_text_stats'] = {
                            "mean_length": sum(field_lengths) / len(field_lengths),
                            "max_length": max(field_lengths),
                            "min_length": min(field_lengths)
                        }
        
        return stats
