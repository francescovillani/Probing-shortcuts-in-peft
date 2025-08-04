from wilds import get_dataset
from datasets import Dataset
from typing import Union
import pandas as pd
import torch

from .base_loader import BaseDatasetLoader

class WildsDatasetLoader(BaseDatasetLoader):
    """Loader for datasets from the WILDS benchmark."""

    def load(self, dataset_name: str, **kwargs) -> Union[Dataset, None]:
        """
        Load a dataset from the WILDS benchmark.

        Args:
            dataset_name: The name of the WILDS dataset to load (e.g., 'civilcomments').
            **kwargs: Additional arguments for get_dataset (e.g., download=True).

        Returns:
            A Hugging Face Dataset object or None if loading fails.
        """
        try:
            # Download and get the dataset from the WILDS library
            wilds_dataset = get_dataset(dataset=dataset_name)
            split_data = wilds_dataset.get_subset(kwargs.get('split', 'train'))

        except Exception as e:
            # logger.error(f"Failed to load dataset '{dataset_name}' from WILDS: {e}")
            raise e

        all_data = []
        for i in range(len(split_data)):
            data_point = split_data[i]
            text, label, metadata = data_point
            
            # Skip examples with None text
            if text is None:
                continue
                
            # Convert tensors to Python native types
            if torch.is_tensor(label):
                label = label.item() if label.numel() == 1 else label.tolist()
            
            # Metadata is a tensor, convert it to a list
            if torch.is_tensor(metadata):
                metadata_list = metadata.tolist()
            else:
                metadata_list = metadata
            
            # For civilcomments, metadata contains information about subgroups
            # We can expand this into columns.
            # metadata[0] is 'male', [1] is 'female', etc.
            # We can name them generically for now
            row = {
                'text': text,
                'label': label,
            }
            for i, val in enumerate(metadata_list):
                 row[f'metadata_{i}'] = val
            all_data.append(row)

        df = pd.DataFrame(all_data)
        
        # Convert the pandas DataFrame to a Hugging Face Dataset
        self.dataset = Dataset.from_pandas(df)
        
        return self.dataset 