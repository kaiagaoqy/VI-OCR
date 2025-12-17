"""
GPT Model Inference Wrapper
"""

from .base_model import BaseModelInference
from openai import OpenAI
from PIL import Image
import base64
import os
import tempfile
from typing import Union
import numpy as np


class GPTModelInference(BaseModelInference):
    """Wrapper for OpenAI GPT models."""
    
    def __init__(self, model_path: str = "gpt-4o", **kwargs):
        """
        Initialize GPT model.
        
        Args:
            model_path: GPT model name (e.g., "gpt-4o", "gpt-4-vision-preview")
        """
        super().__init__(model_path, **kwargs)
        
    def _load_model(self, **kwargs):
        """Load OpenAI client."""
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        print(f"✓ OpenAI client initialized with model: {self.model_path}")
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def infer(self, image: Union[str, np.ndarray, Image.Image], prompt: str, **kwargs) -> str:
        """
        Perform inference using GPT.
        
        Args:
            image: Path to image file, numpy array, or PIL Image
            prompt: Text prompt
            **kwargs: Additional arguments (temperature, max_tokens, etc.)
            
        Returns:
            Model output text
        """
        # Client should be loaded in __init__ via _load_model()
        
        # Convert to file path if needed
        if isinstance(image, str):
            image_path = image
        elif isinstance(image, np.ndarray):
            # Save to temp file
            pil_img = Image.fromarray(image.astype(np.uint8))
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            pil_img.save(temp_file.name)
            image_path = temp_file.name
        elif isinstance(image, Image.Image):
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            image.save(temp_file.name)
            image_path = temp_file.name
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Encode image
        image_base64 = self.encode_image(image_path)
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Call API
        completion = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=kwargs.get('temperature', 0),
            max_tokens=kwargs.get('max_tokens', 1000)
        )
        
        output = completion.choices[0].message.content.strip()
        
        # Clean up temp file if created
        if not isinstance(image, str):
            try:
                os.remove(image_path)
            except:
                pass
        
        return output


