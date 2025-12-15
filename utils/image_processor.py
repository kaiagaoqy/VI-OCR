"""
Image Processor Module
Provides wrapper for image filtering functionality without modifying the original filter.py
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import sys
import os

# Import the original filter function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from filter import add_filter as original_add_filter


class ImageFilterProcessor:
    """
    Wrapper class for image filtering operations.
    Provides a clean interface to apply low vision filters to images.
    """
    
    def __init__(self, 
                 screen_reso: Tuple[int, int] = (2796, 1290),
                 screen_size: float = 13.3,
                 camera: bool = False,
                 white_balance: bool = False,
                 resize: bool = True):
        """
        Initialize the ImageFilterProcessor.
        
        Args:
            screen_reso (tuple): Screen resolution in pixels. Defaults to (2796, 1290).
            screen_size (float): Screen size in inches. Defaults to 13.3.
            camera (bool): Whether the image is taken by a camera. Defaults to False.
            white_balance (bool): Whether to apply white balance. Defaults to False.
            resize (bool): Whether to resize the image to the screen resolution. Defaults to True.
        """
        self.screen_reso = screen_reso
        self.screen_size = screen_size
        self.camera = camera
        self.white_balance = white_balance
        self.resize = resize

    def apply_filter(self, 
                    image: Union[str, np.ndarray, Image.Image],
                    hshift: float,
                    vshift: float) -> np.ndarray:
        """
        Apply low vision filter to an image.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            hshift: Horizontal shift value (for blur effect)
            vshift: Vertical shift value (for contrast effect)
            
        Returns:
            np.ndarray: Filtered image as numpy array in RGB format
        """
        # Load image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img = image.copy()
            
        # Apply the filter using the original function
        filtered_img = original_add_filter(
            img=img,
            HShift=hshift,
            VShift=vshift,
            screen_reso=self.screen_reso,
            screen_size=self.screen_size,
            camera=self.camera,
            white_balance=self.white_balance,
            resize=self.resize
        )
        
        return filtered_img
    
    def apply_filter_to_pil(self,
                           image: Union[str, np.ndarray, Image.Image],
                           hshift: float,
                           vshift: float) -> Image.Image:
        """
        Apply filter and return PIL Image.
        
        Args:
            image: Input image
            hshift: Horizontal shift value
            vshift: Vertical shift value
            
        Returns:
            PIL.Image: Filtered image as PIL Image
        """
        filtered_array = self.apply_filter(image, hshift, vshift)
        return Image.fromarray(filtered_array.astype(np.uint8))
    
    @staticmethod
    def get_standard_hshift_vshift():
        """
        Get the standard HShift and VShift lists used in the original filter.py
        
        Returns:
            tuple: (HShiftList, VShiftList)
        """
        HShiftList = [
            1.000, 0.288, 0.157, 0.086, 0.048, 0.027,
            0.250, 0.134, 0.072, 0.039, 0.022,
            0.267, 0.144, 0.078, 0.043, 0.024,
            0.314, 0.172, 0.096, 0.055, 0.032,
            0.345, 0.193, 0.110, 0.064, 0.038,
            0.439, 0.256, 0.154, 0.033, 0.018,
            0.125, 0.063, 0.031, 0.016, 1.000,
            1.000, 1.000, 1.000, 1.000, 1.000
        ]
        
        VShiftList = [
            1.000, 0.288, 0.157, 0.086, 0.048, 0.027,
            1.000, 0.534, 0.288, 0.157, 0.086,
            0.534, 0.288, 0.157, 0.086, 0.048,
            0.157, 0.086, 0.048, 0.027, 0.016,
            0.086, 0.048, 0.027, 0.016, 0.010,
            0.027, 0.016, 0.010, 0.534, 0.288,
            1.000, 1.000, 1.000, 1.000, 0.355,
            0.178, 0.089, 0.045, 0.022, 0.011
        ]
        
        return HShiftList, VShiftList
    
    @staticmethod
    def convert_filter_id_to_shifts(filter_id: int) -> Tuple[float, float]:
        """
        Convert filter ID to corresponding HShift and VShift values.
        
        Args:
            filter_id: Filter ID (1-based index)
            
        Returns:
            tuple: (hshift, vshift) values
        """
        HShiftList, VShiftList = ImageFilterProcessor.get_standard_hshift_vshift()
        
        if filter_id < 1 or filter_id > len(HShiftList):
            raise ValueError(f"Filter ID must be between 1 and {len(HShiftList)}")
        
        # Convert to internal format (reciprocal for HShift)
        hshift = np.round(1 / HShiftList[filter_id - 1], 4)
        vshift = VShiftList[filter_id - 1]
        
        return hshift, vshift
    
    @staticmethod
    def get_shifts_for_subject(subject_id: str, 
                               param_matrix_path: str = 'data/param.matrix.csv',
                               human_vision_path: str = 'data/human/human_measured_vision_cleaned.csv',
                               verbose: bool = False) -> Tuple[float, float]:
        """
        Get hshift and vshift values for a subject based on their measured VA and CS.
        
        This method uses the VisionParameterLookup tool to find the appropriate
        filter parameters for a subject based on their measured visual acuity and
        contrast sensitivity.
        
        Args:
            subject_id: Subject ID (e.g., 'Sub123')
            param_matrix_path: Path to param.matrix.csv
            human_vision_path: Path to human_measured_vision_cleaned.csv
            verbose: Print detailed information
            
        Returns:
            tuple: (hshift, vshift) values for the subject
            
        Example:
            >>> hshift, vshift = ImageFilterProcessor.get_shifts_for_subject('Sub123')
            >>> filtered_img = processor.apply_filter(img, hshift, vshift)
        """
        try:
            from .vision_parameter_lookup import VisionParameterLookup
        except ImportError:
            # If relative import fails, try absolute
            from vision_parameter_lookup import VisionParameterLookup
        
        lookup = VisionParameterLookup(param_matrix_path, human_vision_path)
        result = lookup.get_filter_params_for_subject(subject_id, verbose=verbose)
        
        return result['hshift'], result['vshift']

