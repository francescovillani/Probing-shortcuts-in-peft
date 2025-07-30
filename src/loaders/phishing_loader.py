"""
Custom dataset loader for phishing/phishing email classification dataset.

This loader handles the phishing email dataset with pickle files containing
email data with various text fields and binary classification labels.
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


class PhishingDatasetLoader(BaseDatasetLoader):
    """
    Custom loader for phishing/phishing email classification dataset.
    
    This loader can work with different pickle files in the phishing dataset:
    - full_dataframe.pkl: Complete dataset with both phishing and benign emails
    - phishing_dataframe.pkl: Only phishing emails
    - enron_dataframe.pkl: Only benign (Enron) emails
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        super().__init__(tokenizer, max_length)
        self.available_text_fields = ['Subject', 'Content', 'clean_subj', 'clean_content']
        self.label_field = 'Tag'
        
    def load(
        self, 
        dataset_path: str,
        pickle_file: str = "full_dataframe.pkl",
        text_field: Union[str, List[str]] = "clean_content",
        split: Optional[str] = None,
        test_size: Optional[float] = None,
        seed: int = 42
    ) -> Dataset:
        """
        Load the phishing dataset from pickle file.
        
        Args:
            dataset_path: Path to the phishing dataset directory
            pickle_file: Name of the pickle file to load (default: full_dataframe.pkl)
            text_field: Text field(s) to use for classification. Can be single field or list.
                       Available: ['Subject', 'Content', 'clean_subj', 'clean_content']
            split: Optional split to return ('train', 'test', or None for full dataset)
            test_size: If split is provided, proportion for test split (default: 0.2)
            seed: Random seed for train/test split
            
        Returns:
            Dataset object ready for training/evaluation
        """
        # Construct full path to pickle file
        if not os.path.isabs(dataset_path):
            # If relative path, assume it's relative to project root
            project_root = Path(__file__).parent.parent.parent
            full_path = project_root / dataset_path / pickle_file
        else:
            full_path = Path(dataset_path) / pickle_file
            
        if not full_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {full_path}")
            
        logger.info(f"Loading phishing dataset from: {full_path}")
        
        # Load the dataframe
        try:
            df = pd.read_pickle(full_path)
            logger.info(f"Loaded dataframe with shape: {df.shape}")
            logger.info(f"Available columns: {list(df.columns)}")
            
        except Exception as e:
            raise ValueError(f"Failed to load pickle file {full_path}: {str(e)}")
            
        # Validate text fields
        if isinstance(text_field, str):
            text_fields = [text_field]
        else:
            text_fields = text_field
            
        missing_fields = [field for field in text_fields if field not in df.columns]
        if missing_fields:
            raise ValueError(
                f"Text field(s) {missing_fields} not found in dataset. "
                f"Available text fields: {[col for col in self.available_text_fields if col in df.columns]}"
            )
            
        # Validate label field
        if self.label_field not in df.columns:
            raise ValueError(f"Label field '{self.label_field}' not found in dataset columns: {list(df.columns)}")
            
        # Log label distribution
        label_counts = df[self.label_field].value_counts()
        logger.info(f"Label distribution: {dict(label_counts)}")
        
        # Prepare the dataset
        dataset_dict = {}
        
        # Add text fields
        for field in text_fields:
            # Clean and prepare text data
            text_data = df[field].fillna("").astype(str)
            # Basic cleaning: remove extra whitespace and newlines
            text_data = text_data.str.replace(r'\s+', ' ', regex=True).str.strip()
            dataset_dict[field] = text_data.tolist()
            
        # Convert labels to numeric (0: Benign, 1: Phish)
        label_mapping = {'Benign': 0, 'Phish': 1}
        numeric_labels = df[self.label_field].map(label_mapping)
        
        if numeric_labels.isnull().any():
            unique_labels = df[self.label_field].unique()
            raise ValueError(f"Unknown labels found: {unique_labels}. Expected: {list(label_mapping.keys())}")
            
        dataset_dict['labels'] = numeric_labels.tolist()
        
        # Add additional metadata fields that might be useful
        if 'msg_id' in df.columns:
            dataset_dict['msg_id'] = df['msg_id'].astype(str).tolist()
        if 'From' in df.columns:
            dataset_dict['from_address'] = df['From'].fillna("").astype(str).tolist()
        if 'from_domain' in df.columns:
            dataset_dict['from_domain'] = df['from_domain'].fillna("").astype(str).tolist()
            
        # Create HuggingFace Dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Convert label column to ClassLabel for proper stratification support
        # This is required for train_test_split with stratify_by_column
        label_names = ["Benign", "Phish"]  # Order matters: 0=Benign, 1=Phish
        dataset = dataset.cast_column("labels", ClassLabel(names=label_names))
        
        logger.info(f"Created dataset with {len(dataset)} examples")
        logger.info(f"Dataset features: {list(dataset.features.keys())}")
        logger.info(f"Label column type: {dataset.features['labels']}")
        
        # Store metadata about the dataset
        dataset._custom_loader_type = 'phishing'
        dataset.label_mapping = label_mapping
        dataset.text_fields = text_fields
        
        self.dataset = dataset
        return dataset
        
    def get_text_fields(self) -> List[str]:
        """Return available text fields for this dataset."""
        return self.available_text_fields
        
    def get_label_mapping(self) -> Dict[str, int]:
        """Return the label mapping used by this dataset."""
        return {'Benign': 0, 'Phish': 1}
        
    def tokenize(self, batch: Dict, text_fields: Union[str, List[str]] = None) -> Dict:
        """
        Tokenize the input batch for phishing dataset.
        
        Overrides the base tokenize method to handle phishing-specific text fields.
        """
        if text_fields is None:
            # Default to clean_content if available, otherwise use Content
            if 'clean_content' in batch:
                text_fields = 'clean_content'
            elif 'Content' in batch:
                text_fields = 'Content'
            elif 'clean_subj' in batch:
                text_fields = 'clean_subj'
            elif 'Subject' in batch:
                text_fields = 'Subject'
            else:
                raise ValueError(f"No suitable text field found in batch. Available fields: {list(batch.keys())}")
        
        # Use parent class tokenization with the determined text fields
        return super().tokenize(batch, text_fields) 