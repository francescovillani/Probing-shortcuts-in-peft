from typing import Dict, Optional
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import pandas as pd
import os
import torch

from .dataset import BaseDatasetLoader, DatasetManager

class SimplePairLoader(BaseDatasetLoader):
    """Loader for the simple pair datasets."""
    
    # Define the label mapping
    LABEL_MAP = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2
    }
    
    def load(self, path: str, split: Optional[str] = None) -> DatasetDict:
        """
        Load the simple pair dataset from a directory containing CSV files.
        
        Args:
            path: Path to the directory containing the CSV files
            split: Optional split name to load a specific file
            
        Returns:
            DatasetDict containing the loaded datasets
        """
        datasets = {}
        
        if split:
            # Load specific split
            file_path = os.path.join(path, f"{split}.csv")
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            df = pd.read_csv(file_path, sep='\t', header=None, 
                           names=['text1', 'text2', 'label'])
            datasets[split] = Dataset.from_pandas(df)
        else:
            # Load all CSV files in the directory
            for file_name in os.listdir(path):
                if file_name.endswith('.csv'):
                    split_name = os.path.splitext(file_name)[0]
                    file_path = os.path.join(path, file_name)
                    df = pd.read_csv(file_path, sep='\t', header=None,
                                   names=['text1', 'text2', 'label'])
                    datasets[split_name] = Dataset.from_pandas(df)
        
        return DatasetDict(datasets)
    
    def tokenize(self, batch: Dict, text_fields: Optional[str] = None) -> Dict:
        """
        Tokenize the text pairs in the batch.
        
        Args:
            batch: Batch of examples
            text_fields: Not used, as we always use text1 and text2
            
        Returns:
            Dict containing tokenized inputs
        """
        # Tokenize both text fields with special tokens
        tokenized = self.tokenizer(
            batch['text1'],
            batch['text2'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert string labels to integers using the mapping
        labels = [self.LABEL_MAP[label] for label in batch['label']]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels_tensor
        }

# Register the loader with DatasetManager
DatasetManager.register_loader("simple_pair", SimplePairLoader) 