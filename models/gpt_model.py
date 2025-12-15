"""
GPT Model Inference Wrapper
"""

from .base_model import BaseModelInference
from openai import OpenAI
import base64
import os


class GPTModelInference(BaseModelInference):
    """Wrapper for OpenAI GPT models."""
    
    def __init__(self, model_path: str = "gpt-4o"):
        """
        Initialize GPT model.
        
        Args:
            model_path: GPT model name (e.g., "gpt-4o", "gpt-4-vision-preview")
        """
        super().__init__(model_path)
        self.model_name = model_path
        
    def load_model(self):
        """Load OpenAI client."""
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        print(f"✓ OpenAI client initialized with model: {self.model_name}")
    
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
    
    def infer(self, image: str, prompt: str, **kwargs) -> str:
        """
        Perform inference using GPT.
        
        Args:
            image: Path to image file
            prompt: Text prompt
            **kwargs: Additional arguments (temperature, max_tokens, etc.)
            
        Returns:
            Model output text
        """
        if not hasattr(self, 'client'):
            self.load_model()
        
        # Encode image
        image_base64 = self.encode_image(image)
        
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
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get('temperature', 0),
            max_tokens=kwargs.get('max_tokens', 1000)
        )
        
        output = completion.choices[0].message.content.strip()
        return output

