"""
DeepSeek OCR Model Inference Module
Implements inference for DeepSeek-OCR models.
"""

import torch
from typing import Union
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base_model import BaseModelInference


class DeepSeekOCRInference(BaseModelInference):
    """
    DeepSeek-OCR model inference implementation.
    """
    
    def __init__(self, model_path: str, size: str = "Gundam", **kwargs):
        """
        Initialize DeepSeek OCR model.
        
        Args:
            model_path: Path to the model
            size: Model size configuration (Tiny, Small, Base, Large, Gundam)
            **kwargs: Additional parameters
        """
        self.size = size
        self.size_dict = {
            "Tiny": (512, 512, False),
            "Small": (640, 640, False),
            "Base": (1024, 1024, False),
            "Large": (1280, 1280, False),
            "Gundam": (1024, 640, True)
        }
        super().__init__(model_path, **kwargs)
    
    def _load_model(self, **kwargs):
        """
        Load DeepSeek-OCR model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            use_safetensors=True
        )
        self.model = self.model.eval().cuda().to(torch.bfloat16)
    
    def infer(self, 
             image: Union[str, np.ndarray, Image.Image], 
             prompt: str = "<image>\nFree OCR. ",
             **kwargs) -> str:
        """
        Perform inference on an image.
        
        Args:
            image: Input image
            prompt: Text prompt for the model (default: "<image>\nFree OCR. ")
            **kwargs: Additional parameters
            
        Returns:
            str: Model output text
        """
        # Convert image to file path if needed
        # DeepSeek-OCR model expects image file path
        if isinstance(image, str):
            image_file = image
        elif isinstance(image, np.ndarray):
            # Save temporarily
            import tempfile
            pil_image = Image.fromarray(image.astype(np.uint8))
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            pil_image.save(temp_file.name)
            image_file = temp_file.name
        elif isinstance(image, Image.Image):
            # Save temporarily
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            image.save(temp_file.name) # Save the image to a temporary file
            image_file = temp_file.name
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Get size configuration
        base_size, image_size, crop_mode = self.size_dict[self.size]
        
        # Perform inference
        output = self.model.infer(
            self.tokenizer, 
            prompt=prompt, 
            image_file=image_file, 
            base_size=base_size, # Base size of the image
            image_size=image_size, # Image size of the image
            crop_mode=crop_mode, # Crop mode of the image
            save_results=False, # Save the results to a file
            test_compress=True, # Test compress the image
            output_path=' ', # Output path of the results
            eval_mode=True # Evaluation mode
        )
        
        # Cleanup
        torch.cuda.empty_cache()
        
        # Clean up temporary file if created
        if not isinstance(image, str):
            import os
            try:
                os.remove(image_file)
            except:
                pass
        
        return output.strip() if isinstance(output, str) else output

