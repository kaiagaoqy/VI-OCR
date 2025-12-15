"""
CogVLM Model Inference Wrapper
"""

from .base_model import BaseModelInference
from PIL import Image
import torch
import os
from transformers import AutoModelForCausalLM, LlamaTokenizer


class CogVLMModelInference(BaseModelInference):
    """Wrapper for CogVLM models."""
    
    def __init__(self, model_path: str = "THUDM/cogagent-chat-hf", 
                 tokenizer_path: str = "lmsys/vicuna-7b-v1.5",
                 quant: int = 4,
                 fp16: bool = False,
                 bf16: bool = False):
        """
        Initialize CogVLM model.
        
        Args:
            model_path: Path to CogVLM model
            tokenizer_path: Path to tokenizer
            quant: Quantization bits (4 or None)
            fp16: Use FP16 precision
            bf16: Use BF16 precision
        """
        super().__init__(model_path)
        self.tokenizer_path = tokenizer_path
        self.quant = quant
        self.fp16 = fp16
        self.bf16 = bf16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load CogVLM model and tokenizer."""
        print(f"Loading CogVLM model from {self.model_path}...")
        
        # Load tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_path)
        
        # Determine torch type
        if self.bf16:
            torch_type = torch.bfloat16
        else:
            torch_type = torch.float16
        
        self.torch_type = torch_type
        
        # Load model
        if self.quant:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_type,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                device_map="auto"
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_type,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto"
            ).to(self.device).eval()
        
        print(f"✓ CogVLM model loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  Torch type: {torch_type}")
        print(f"  Quantization: {self.quant}-bit" if self.quant else "  No quantization")
    
    def infer(self, image: str, prompt: str, **kwargs) -> str:
        """
        Perform inference using CogVLM.
        
        Args:
            image: Path to image file
            prompt: Text prompt
            **kwargs: Additional arguments (max_length, do_sample, etc.)
            
        Returns:
            Model output text
        """
        if not hasattr(self, 'model'):
            self.load_model()
        
        # Load image
        img = Image.open(image).convert('RGB')
        
        # Build conversation input
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=prompt,
            history=[],
            images=[img]
        )
        
        # Prepare inputs
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[input_by_model['images'][0].to(self.device).to(self.torch_type)]],
        }
        
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.device).to(self.torch_type)]]
        
        # Generation parameters
        gen_kwargs = {
            "max_length": kwargs.get('max_length', 2048),
            "do_sample": kwargs.get('do_sample', False),
            "use_cache": kwargs.get('use_cache', True)
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            output = self.tokenizer.decode(outputs[0])
            output = output.split("</s>")[0].strip().replace(".", "")
        
        return output

