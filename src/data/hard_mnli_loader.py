import pandas as pd
from datasets import Dataset
import os
import logging
from .dataset import BaseDatasetLoader, DatasetManager


logger = logging.getLogger(__name__)

class HardMNLILoader(BaseDatasetLoader):
    """Loader for the filtered hard MNLI dataset."""
    
    def load(self, dataset_path: str, split: str = None) -> Dataset:
        """Load the hard MNLI dataset from TSV file.
        
        Args:
            dataset_path: Path to the directory containing the TSV files
            split: Split to load (e.g., 'dev_matched_hard')
        """
        logger.info(f"Loading Hard MNLI dataset from {dataset_path}")
        
        # Construct the full path
        if split:
            file_path = os.path.join(dataset_path, f"{split}.tsv")
        else:
            file_path = dataset_path
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        # Read the TSV file
        df = pd.read_csv(file_path, sep='\t', header=None)
        
        # Map the columns based on MNLI format
        # Adjust these column names based on your TSV structure
        columns = {
            8: 'premise',
            9: 'hypothesis',
            10: 'label'  # Assuming the label is in column 10
        }
        
        # Create a new dataframe with only the columns we need
        filtered_df = df[columns.keys()].rename(columns=columns)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(filtered_df)
        
        logger.info(f"Loaded {len(dataset)} examples from {file_path}")
        return dataset 
    
# Register the HardMNLI loader at the end of the file
DatasetManager.register_loader("hard_mnli", HardMNLILoader) 