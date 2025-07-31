"""
Custom dataset loaders for various datasets.

This module contains custom dataset loaders that extend the base functionality
provided by the DatasetService.
"""

from .base_loader import BaseDatasetLoader
from .phishing_loader import PhishingDatasetLoader

# Registry of all custom loaders
CUSTOM_LOADERS = {
    'phishing': PhishingDatasetLoader,  # Alias for phishing
}

def register_custom_loaders(dataset_service):
    """
    Register all custom loaders with the dataset service.
    
    Args:
        dataset_service: DatasetService instance to register loaders with
    """
    for loader_name, loader_class in CUSTOM_LOADERS.items():
        dataset_service.register_loader(loader_name, loader_class)

__all__ = ['BaseDatasetLoader', 'PhishingDatasetLoader', 'CUSTOM_LOADERS', 'register_custom_loaders'] 