"""
Qwen2 Model Inference Module
Implements inference for Qwen2.5-VL models.
"""

import torch
from typing import Union, Optional
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from .base_model import BaseModelInference


class Qwen2ModelInference(BaseModelInference):
    """
    Qwen2.5-VL model inference implementation.
    """
    
    def _load_model(self, **kwargs):
        """
        Load Qwen2.5-VL model and processor.
        """
        torch_dtype = kwargs.get('torch_dtype', 'auto')
        device_map = kwargs.get('device_map', 'auto')
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
    def infer(self, 
             image: Union[str, np.ndarray, Image.Image], 
             prompt: str,
             max_new_tokens: int = 128,
             **kwargs) -> str:
        """
        Perform inference on an image.
        
        Args:
            image: Input image
            prompt: Text prompt for the model
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            str: Model output text
        """
        # Convert image to appropriate format
        if isinstance(image, str):
            image_path = image
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image and save temporarily
            pil_image = Image.fromarray(image.astype(np.uint8))
            image_path = pil_image
        elif isinstance(image, Image.Image):
            image_path = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Prepare messages in Qwen format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        # Cleanup
        del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
        torch.cuda.empty_cache()
        
        return output[0].strip()

