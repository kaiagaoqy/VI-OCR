"""
Gemini Model Inference Module
Implements inference for Google Gemini models.
"""

import os
from typing import Union
import numpy as np
from PIL import Image
from google import genai
from retry import retry
from dotenv import load_dotenv

from .base_model import BaseModelInference

load_dotenv(".env")


class GeminiModelInference(BaseModelInference):
    """
    Google Gemini model inference implementation.
    """
    
    def _load_model(self, **kwargs):
        """
        Load Gemini client.
        """
        api_key = kwargs.get('api_key', os.environ.get("Gemini-API-KEY"))
        if not api_key:
            raise ValueError("Gemini API key not found. Set 'Gemini-API-KEY' in environment or pass as parameter.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name_id = self.model_path  # e.g., "gemini-1.5-flash"
        
    @retry((Exception), tries=3, delay=0, backoff=0)
    def _call_gemini(self, messages, parse_fn=None):
        """
        Call Gemini API with retry logic.
        """
        response = self.client.models.generate_content(
            model=self.model_name_id,
            contents=messages,
        )
        ret = ""
        try:
            if parse_fn is not None:
                ret = parse_fn(response.text)
            else:
                ret = response.text
        except:
            print(response.prompt_feedback)
        return ret
    
    def infer(self, 
             image: Union[str, np.ndarray, Image.Image], 
             prompt: str,
             parse_fn=None,
             **kwargs) -> str:
        """
        Perform inference on an image.
        
        Args:
            image: Input image
            prompt: Text prompt for the model
            parse_fn: Optional function to parse the response
            **kwargs: Additional parameters
            
        Returns:
            str: Model output text
        """
        # Convert image to PIL Image if needed
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Prepare messages in Gemini format
        messages = [pil_image, prompt]
        
        # Call Gemini API
        output = self._call_gemini(messages, parse_fn=parse_fn)
        
        return output.strip()

