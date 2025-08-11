"""
Custom dataset loaders for various datasets.

This module contains custom dataset loaders that extend the base functionality
provided by the DatasetService.
"""

from .base_loader import BaseDatasetLoader
from .phishing_loader import PhishingDatasetLoader
from .isot_loader import IsotDatasetLoader
from .liar_loader import LiarDatasetLoader
from .fake_reviews_loader import FakeReviewsDatasetLoader

# Registry of all custom loaders
CUSTOM_LOADERS = {
    'phishing': PhishingDatasetLoader,  # Alias for phishing
    'isot': IsotDatasetLoader,          # ISOT Fake News dataset
    'isot_fake_news': IsotDatasetLoader, # Alternative alias for ISOT
    'liar': LiarDatasetLoader,          # LIAR dataset
    'liar_dataset': LiarDatasetLoader,  # Alternative alias for LIAR
    'fake_reviews': FakeReviewsDatasetLoader,  # Alternative alias for LIAR
    'fake_reviews_dataset': FakeReviewsDatasetLoader,  # Alternative alias for LIAR
}

def register_custom_loaders(service):
    """
    Registers all custom dataset loaders with the DatasetService.
    
    Args:
        service: DatasetService instance to register loaders with
    """
    for loader_name, loader_class in CUSTOM_LOADERS.items():
        service.register_loader(loader_name, loader_class)

__all__ = ['BaseDatasetLoader', 'PhishingDatasetLoader', 'IsotDatasetLoader', 'LiarDatasetLoader', 'CUSTOM_LOADERS', 'register_custom_loaders'] 