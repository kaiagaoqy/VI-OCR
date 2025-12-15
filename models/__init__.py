"""
Models Package
Provides unified interface for all model inference implementations.

Uses lazy imports to avoid loading all model dependencies at once.
This allows using different models in different virtual environments.
"""

from .base_model import BaseModelInference

__all__ = [
    'BaseModelInference',
    'ModelFactory'
]


class ModelFactory:
    """
    Factory class to create model instances based on model type.
    Uses lazy imports to avoid dependency conflicts.
    """
    
    # Map model types to (module_name, class_name)
    MODEL_MAP = {
        'qwen2': ('qwen2_model', 'Qwen2ModelInference'),
        'qwen2.5': ('qwen2_model', 'Qwen2ModelInference'),
        'gemini': ('gemini_model', 'GeminiModelInference'),
        'dsocr': ('dsocr_model', 'DeepSeekOCRInference'),
        'deepseek-ocr': ('dsocr_model', 'DeepSeekOCRInference'),
        'claude': ('claude_model', 'ClaudeModelInference'),
        'gpt': ('gpt_model', 'GPTModelInference'),
        'gpt4': ('gpt_model', 'GPTModelInference'),
        'gpt-4': ('gpt_model', 'GPTModelInference'),
        'cogvlm': ('cogvlm_model', 'CogVLMModelInference'),
    }
    
    @classmethod
    def _lazy_import_model_class(cls, module_name: str, class_name: str):
        """
        Lazily import a model class only when needed.
        
        Args:
            module_name: Name of the module (e.g., 'qwen2_model')
            class_name: Name of the class (e.g., 'Qwen2ModelInference')
            
        Returns:
            The model class
            
        Raises:
            ImportError: If the module or class cannot be imported
        """
        try:
            # Import the module dynamically
            module = __import__(f'models.{module_name}', fromlist=[class_name])
            model_class = getattr(module, class_name)
            return model_class
        except ImportError as e:
            raise ImportError(
                f"Failed to import {class_name} from models.{module_name}. "
                f"Make sure the required dependencies are installed in your environment. "
                f"Original error: {e}"
            )
    
    @classmethod
    def create_model(cls, model_type: str, model_path: str, **kwargs) -> BaseModelInference:
        """
        Create a model instance based on model type.
        
        Args:
            model_type: Type of model (qwen2, gemini, dsocr, claude, gpt, cogvlm)
            model_path: Path to the model or model identifier
            **kwargs: Additional model-specific parameters
            
        Returns:
            BaseModelInference: Model instance
            
        Raises:
            ValueError: If model type is not supported
            ImportError: If model dependencies are not installed
        """
        model_type_lower = model_type.lower()
        
        if model_type_lower not in cls.MODEL_MAP:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(cls.MODEL_MAP.keys())}"
            )
        
        # Lazy import the model class
        module_name, class_name = cls.MODEL_MAP[model_type_lower]
        model_class = cls._lazy_import_model_class(module_name, class_name)
        
        # Create and return the model instance
        return model_class(model_path, **kwargs)
    
    @classmethod
    def register_model(cls, model_type: str, module_name: str, class_name: str):
        """
        Register a new model type with lazy loading.
        
        Args:
            model_type: Type identifier for the model
            module_name: Name of the module containing the model class
            class_name: Name of the model class
        """
        cls.MODEL_MAP[model_type.lower()] = (module_name, class_name)

