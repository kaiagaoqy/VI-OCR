"""
Claude Model Inference Wrapper
"""

from .base_model import BaseModelInference
from PIL import Image
import anthropic
import base64
import os
import tempfile
from io import BytesIO
from typing import Union
import numpy as np


class ClaudeModelInference(BaseModelInference):
    """Wrapper for Claude API models."""
    
    def __init__(self, model_path: str = "claude-3-5-sonnet-20240620", **kwargs):
        """
        Initialize Claude model.
        
        Args:
            model_path: Claude model name (e.g., "claude-3-5-sonnet-20240620")
        """
        super().__init__(model_path, **kwargs)
        
    def _load_model(self, **kwargs):
        """Load Claude client."""
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        print(f"✓ Claude client initialized with model: {self.model_path}")
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 with size check.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            img_byte = image_file.read()
            img_size = img_byte.__sizeof__()
            
            # Claude has a 5MB limit, resize if needed
            if img_size >= 5242880:
                img = Image.open(image_path).convert('RGB')
                w, h = img.size
                ratio = 1568 / np.max([w, h])
                img = img.resize((int(w * ratio), int(h * ratio)))
                
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte = img_byte_arr.getvalue()
            
            return base64.b64encode(img_byte).decode('utf-8')
    
    def infer(self, image: Union[str, np.ndarray, Image.Image], prompt: str, **kwargs) -> str:
        """
        Perform inference using Claude.
        
        Args:
            image: Path to image file, numpy array, or PIL Image
            prompt: Text prompt
            **kwargs: Additional arguments (max_tokens, temperature, etc.)
            
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
        
        # Determine media type
        postfix = image_path.split(".")[-1].lower()
        
        if postfix == "png":
            media_type = "image/png"
        elif postfix in ["jpg", "jpeg"]:
            media_type = "image/jpeg"
        else:
            media_type = "image/png"  # default
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ]
        
        # Call API
        max_tokens = kwargs.get('max_tokens', 1000)
        temperature = kwargs.get('temperature', 0)
        
        response = self.client.messages.create(
            model=self.model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are a helpful AI assistant.",
            messages=messages
        )
        
        output = response.content[0].text.strip()
        
        # Clean up temp file if created
        if not isinstance(image, str):
            try:
                os.remove(image_path)
            except:
                pass
        
        return output


