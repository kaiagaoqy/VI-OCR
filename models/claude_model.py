"""
Claude Model Inference Wrapper
"""

from .base_model import BaseModelInference
from PIL import Image
import anthropic
import base64
import os
from io import BytesIO
import numpy as np


class ClaudeModelInference(BaseModelInference):
    """Wrapper for Claude API models."""
    
    def __init__(self, model_path: str = "claude-3-5-sonnet-20240620"):
        """
        Initialize Claude model.
        
        Args:
            model_path: Claude model name (e.g., "claude-3-5-sonnet-20240620")
        """
        super().__init__(model_path)
        self.model_name = model_path
        
    def load_model(self):
        """Load Claude client."""
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        print(f"✓ Claude client initialized with model: {self.model_name}")
    
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
    
    def infer(self, image: str, prompt: str, **kwargs) -> str:
        """
        Perform inference using Claude.
        
        Args:
            image: Path to image file
            prompt: Text prompt
            **kwargs: Additional arguments (max_tokens, temperature, etc.)
            
        Returns:
            Model output text
        """
        if not hasattr(self, 'client'):
            self.load_model()
        
        # Encode image
        image_base64 = self.encode_image(image)
        
        # Determine media type from file extension
        postfix = image.split(".")[-1].lower()
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
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are a helpful AI assistant.",
            messages=messages
        )
        
        output = response.content[0].text.strip()
        return output

