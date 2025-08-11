"""
Custom dataset loader for ISOT Fake News dataset.

This loader handles the ISOT Fake News dataset with separate True.csv and Fake.csv files,
combining them into a single binary classification dataset.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import Dataset, ClassLabel
from transformers import PreTrainedTokenizer
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from .base_loader import BaseDatasetLoader

logger = logging.getLogger(__name__)


class IsotDatasetLoader(BaseDatasetLoader):
    """
    Custom loader for ISOT Fake News dataset.
    
    This loader handles the ISOT dataset which consists of:
    - True.csv: Real news articles
    - Fake.csv: Fake news articles
    
    Each CSV file contains columns: title, text, subject, date
    The loader combines them and adds binary labels (0=True, 1=Fake).
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        super().__init__(tokenizer, max_length)
        self.available_text_fields = ['title', 'text', 'combined']
        self.label_field = 'labels'
        
    def load(
        self, 
        dataset_path: str,
        text_field: Union[str, List[str]] = "combined",
        split: Optional[str] = None,
        test_size: Optional[float] = None,
        seed: int = 42,
        combine_title_text: bool = True,
        title_separator: str = " "
    ) -> Dataset:
        """
        Load the ISOT Fake News dataset from CSV files.
        
        Args:
            dataset_path: Path to the ISOT dataset directory containing True.csv and Fake.csv
            text_field: Text field(s) to use for classification. 
                       Available: ['title', 'text', 'combined'] or list of fields
                       'combined' merges title and text with separator
            split: Optional split to return ('train', 'test', or None for full dataset)
            test_size: If split is provided, proportion for test split (default: 0.2)
            seed: Random seed for train/test split
            combine_title_text: Whether to create a 'combined' field from title + text
            title_separator: Separator used when combining title and text
            
        Returns:
            Dataset object ready for training/evaluation
        """
        # Construct full paths to CSV files
        if not os.path.isabs(dataset_path):
            # If relative path, assume it's relative to project root
            project_root = Path(__file__).parent.parent.parent
            dataset_dir = project_root / dataset_path
        else:
            dataset_dir = Path(dataset_path)
            
        true_path = dataset_dir / "True.csv"
        fake_path = dataset_dir / "Fake.csv"
        
        # Check if files exist
        if not true_path.exists():
            raise FileNotFoundError(f"True.csv not found: {true_path}")
        if not fake_path.exists():
            raise FileNotFoundError(f"Fake.csv not found: {fake_path}")
            
        logger.info(f"Loading ISOT dataset from: {dataset_dir}")
        
        # Load the dataframes
        try:
            true_df = pd.read_csv(true_path)
            fake_df = pd.read_csv(fake_path)
            logger.info(f"Loaded {len(true_df)} true news articles")
            logger.info(f"Loaded {len(fake_df)} fake news articles")
            
        except Exception as e:
            raise ValueError(f"Failed to load CSV files from {dataset_dir}: {str(e)}")
            
        # Validate required columns
        required_columns = ['title', 'text', 'subject', 'date']
        for df, name in [(true_df, 'True.csv'), (fake_df, 'Fake.csv')]:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in {name}: {missing_columns}")
                
        # Add labels
        true_df['labels'] = 0  # True news
        fake_df['labels'] = 1  # Fake news
        
        # Combine datasets
        combined_df = pd.concat([true_df, fake_df], ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df)} articles")
        
        # Log label distribution
        label_counts = combined_df['labels'].value_counts()
        logger.info(f"Label distribution - True: {label_counts[0]} ({label_counts[0]/len(combined_df)*100:.1f}%), "
                   f"Fake: {label_counts[1]} ({label_counts[1]/len(combined_df)*100:.1f}%)")
        
        # Log subject distribution
        subject_counts = combined_df['subject'].value_counts()
        logger.info(f"Subject distribution: {dict(subject_counts.head())}")
        
        # Prepare the dataset
        dataset_dict = {}
        
        # Clean and prepare text data
        combined_df['title'] = combined_df['title'].fillna("").astype(str)
        combined_df['text'] = combined_df['text'].fillna("").astype(str)
        
        # Basic text cleaning
        combined_df['title'] = combined_df['title'].str.replace(r'\s+', ' ', regex=True).str.strip()
        combined_df['text'] = combined_df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Add individual text fields
        dataset_dict['title'] = combined_df['title'].tolist()
        dataset_dict['text'] = combined_df['text'].tolist()
        
        # Create combined field if requested
        if combine_title_text:
            combined_text = (combined_df['title'] + title_separator + combined_df['text']).str.strip()
            dataset_dict['combined'] = combined_text.tolist()
            logger.info(f"Created 'combined' text field using separator: '{title_separator}'")
        
        # Add labels
        dataset_dict['labels'] = combined_df['labels'].tolist()
        
        # Add metadata fields
        dataset_dict['subject'] = combined_df['subject'].astype(str).tolist()
        dataset_dict['date'] = combined_df['date'].astype(str).tolist()
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Convert label column to ClassLabel for proper stratification support
        label_names = ["True", "Fake"]  # Order matters: 0=True, 1=Fake
        dataset = dataset.cast_column("labels", ClassLabel(names=label_names))
        
        logger.info(f"Created dataset with {len(dataset)} examples")
        logger.info(f"Dataset features: {list(dataset.features.keys())}")
        logger.info(f"Available text fields: {[field for field in self.available_text_fields if field in dataset.column_names]}")
        
        # Store metadata about the dataset
        dataset._custom_loader_type = 'isot'
        dataset.label_mapping = {'True': 0, 'Fake': 1}
        dataset.text_fields = text_field if isinstance(text_field, list) else [text_field]
        
        self.dataset = dataset
        return dataset
        
    def get_text_fields(self) -> List[str]:
        """Return available text fields for this dataset."""
        return self.available_text_fields
        
    def get_label_mapping(self) -> Dict[str, int]:
        """Return the label mapping used by this dataset."""
        return {'True': 0, 'Fake': 1}
        
    def tokenize(self, batch: Dict, text_fields: Union[str, List[str]] = None) -> Dict:
        """
        Tokenize the input batch for ISOT dataset.
        
        Overrides the base tokenize method to handle ISOT-specific text fields.
        """
        if text_fields is None:
            # Default to combined if available, otherwise use text
            if 'combined' in batch:
                text_fields = 'combined'
            elif 'text' in batch:
                text_fields = 'text'
            elif 'title' in batch:
                text_fields = 'title'
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
            
        stats = {
            "total_samples": len(self.dataset),
            "label_distribution": dict(self.dataset['labels']),
            "features": list(self.dataset.features.keys()),
            "text_fields": self.available_text_fields
        }
        
        # Calculate text length statistics for available text fields
        if 'combined' in self.dataset.column_names:
            combined_lengths = [len(text) for text in self.dataset['combined']]
            stats['combined_text_stats'] = {
                "mean_length": sum(combined_lengths) / len(combined_lengths),
                "max_length": max(combined_lengths),
                "min_length": min(combined_lengths)
            }
        
        return stats 