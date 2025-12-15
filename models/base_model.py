"""
Base Model Inference Module
Defines the abstract base class for all model inference implementations.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional
import numpy as np
from PIL import Image
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_processor import ImageFilterProcessor


class BaseModelInference(ABC):
    """
    Abstract base class for model inference.
    All model implementations should inherit from this class.
    """
    
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the model.
        
        Args:
            model_path: Path to the model or model identifier
            **kwargs: Additional model-specific parameters
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load_model(**kwargs)
        
    @abstractmethod
    def _load_model(self, **kwargs):
        """
        Load the model. Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def infer(self, image: Union[str, np.ndarray, Image.Image], prompt: str, **kwargs) -> str:
        """
        Perform inference on an image.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            prompt: Text prompt for the model
            **kwargs: Additional inference parameters
            
        Returns:
            str: Model output text
        """
        pass
    
    def infer_with_filter(self,
                         image: Union[str, np.ndarray, Image.Image],
                         prompt: str,
                         hshift: float,
                         vshift: float,
                         filter_processor: Optional[ImageFilterProcessor] = None,
                         **kwargs) -> str:
        """
        Apply filter to image and then perform inference.
        
        Args:
            image: Input image
            prompt: Text prompt for the model
            hshift: Horizontal shift value for filter
            vshift: Vertical shift value for filter
            filter_processor: Optional ImageFilterProcessor instance. If None, creates default one.
            **kwargs: Additional inference parameters
            
        Returns:
            str: Model output text
        """
        # Create filter processor if not provided
        if filter_processor is None:
            filter_processor = ImageFilterProcessor()
        
        # Apply filter to image
        filtered_image = filter_processor.apply_filter(image, hshift, vshift)
        
        # Convert to PIL Image for model input
        if isinstance(filtered_image, np.ndarray):
            filtered_image = Image.fromarray(filtered_image.astype(np.uint8))
        
        # Perform inference on filtered image
        return self.infer(filtered_image, prompt, **kwargs)
    
    def cleanup(self):
        """
        Cleanup resources (e.g., GPU memory).
        Can be overridden by subclasses for specific cleanup needs.
        """
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @property
    def model_name(self) -> str:
        """
        Get the model name.
        """
        return self.model_path.split('/')[-1]

