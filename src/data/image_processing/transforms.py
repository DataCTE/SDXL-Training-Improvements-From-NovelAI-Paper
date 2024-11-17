"""
Image transformation utilities for SDXL training pipeline.

This module provides a collection of image transformations commonly used
in the training pipeline, with support for both PIL Images and tensors.
"""

import logging
import random
from typing import Optional, Tuple, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

from src.data.image_processing.validation import validate_image, validate_tensor

logger = logging.getLogger(__name__)


def resize_image(
    image: Union[Image.Image, torch.Tensor],
    size: Tuple[int, int],
    method: str = 'bilinear'
) -> Union[Image.Image, torch.Tensor]:
    """Resize image to target size.
    
    Args:
        image: Input image (PIL or tensor)
        size: Target size as (height, width)
        method: Interpolation method
        
    Returns:
        Resized image
    """
    if isinstance(image, Image.Image):
        validate_image(image)
        return image.resize(size[::-1], getattr(Image, method.upper()))
    else:
        validate_tensor(image)
        return TF.resize(image, size, antialias=True)


def random_crop(
    image: Union[Image.Image, torch.Tensor],
    size: Tuple[int, int]
) -> Union[Image.Image, torch.Tensor]:
    """Randomly crop image to target size.
    
    Args:
        image: Input image (PIL or tensor)
        size: Target size as (height, width)
        
    Returns:
        Cropped image
    """
    if isinstance(image, Image.Image):
        validate_image(image)
        i = random.randint(0, image.height - size[0])
        j = random.randint(0, image.width - size[1])
        return image.crop((j, i, j + size[1], i + size[0]))
    else:
        validate_tensor(image)
        return TF.crop(image, i, j, size[0], size[1])


def center_crop(
    image: Union[Image.Image, torch.Tensor],
    size: Tuple[int, int]
) -> Union[Image.Image, torch.Tensor]:
    """Center crop image to target size.
    
    Args:
        image: Input image (PIL or tensor)
        size: Target size as (height, width)
        
    Returns:
        Cropped image
    """
    if isinstance(image, Image.Image):
        validate_image(image)
        return TF.center_crop(image, size)
    else:
        validate_tensor(image)
        return TF.center_crop(image, size)


def random_flip(
    image: Union[Image.Image, torch.Tensor],
    p: float = 0.5
) -> Union[Image.Image, torch.Tensor]:
    """Randomly flip image horizontally.
    
    Args:
        image: Input image (PIL or tensor)
        p: Probability of flipping
        
    Returns:
        Flipped image
    """
    if random.random() < p:
        if isinstance(image, Image.Image):
            validate_image(image)
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            validate_tensor(image)
            return TF.hflip(image)
    return image


def random_rotate(
    image: Union[Image.Image, torch.Tensor],
    angle_range: Tuple[float, float],
    p: float = 0.5
) -> Union[Image.Image, torch.Tensor]:
    """Randomly rotate image within angle range.
    
    Args:
        image: Input image (PIL or tensor)
        angle_range: (min_angle, max_angle) in degrees
        p: Probability of rotating
        
    Returns:
        Rotated image
    """
    if random.random() < p:
        angle = random.uniform(*angle_range)
        if isinstance(image, Image.Image):
            validate_image(image)
            return image.rotate(angle, expand=True)
        else:
            validate_tensor(image)
            return TF.rotate(image, angle)
    return image


def adjust_brightness(
    image: Union[Image.Image, torch.Tensor],
    factor: float
) -> Union[Image.Image, torch.Tensor]:
    """Adjust image brightness.
    
    Args:
        image: Input image (PIL or tensor)
        factor: Brightness adjustment factor
        
    Returns:
        Adjusted image
    """
    if isinstance(image, Image.Image):
        validate_image(image)
        return TF.adjust_brightness(image, factor)
    else:
        validate_tensor(image)
        return TF.adjust_brightness(image, factor)


def adjust_contrast(
    image: Union[Image.Image, torch.Tensor],
    factor: float
) -> Union[Image.Image, torch.Tensor]:
    """Adjust image contrast.
    
    Args:
        image: Input image (PIL or tensor)
        factor: Contrast adjustment factor
        
    Returns:
        Adjusted image
    """
    if isinstance(image, Image.Image):
        validate_image(image)
        return TF.adjust_contrast(image, factor)
    else:
        validate_tensor(image)
        return TF.adjust_contrast(image, factor)


def adjust_saturation(
    image: Union[Image.Image, torch.Tensor],
    factor: float
) -> Union[Image.Image, torch.Tensor]:
    """Adjust image saturation.
    
    Args:
        image: Input image (PIL or tensor)
        factor: Saturation adjustment factor
        
    Returns:
        Adjusted image
    """
    if isinstance(image, Image.Image):
        validate_image(image)
        return TF.adjust_saturation(image, factor)
    else:
        validate_tensor(image)
        return TF.adjust_saturation(image, factor)


def random_jitter(
    image: Union[Image.Image, torch.Tensor],
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    p: float = 0.5
) -> Union[Image.Image, torch.Tensor]:
    """Apply random color jittering.
    
    Args:
        image: Input image (PIL or tensor)
        brightness: Max brightness adjustment
        contrast: Max contrast adjustment
        saturation: Max saturation adjustment
        hue: Max hue adjustment
        p: Probability of applying jitter
        
    Returns:
        Jittered image
    """
    if random.random() < p:
        if isinstance(image, Image.Image):
            validate_image(image)
            return TF.adjust_brightness(
                TF.adjust_contrast(
                    TF.adjust_saturation(
                        TF.adjust_hue(
                            image,
                            random.uniform(-hue, hue)
                        ),
                        1 + random.uniform(-saturation, saturation)
                    ),
                    1 + random.uniform(-contrast, contrast)
                ),
                1 + random.uniform(-brightness, brightness)
            )
        else:
            validate_tensor(image)
            return TF.adjust_brightness(
                TF.adjust_contrast(
                    TF.adjust_saturation(
                        TF.adjust_hue(
                            image,
                            random.uniform(-hue, hue)
                        ),
                        1 + random.uniform(-saturation, saturation)
                    ),
                    1 + random.uniform(-contrast, contrast)
                ),
                1 + random.uniform(-brightness, brightness)
            )
    return image


def gaussian_blur(
    image: Union[Image.Image, torch.Tensor],
    kernel_size: int,
    sigma: Optional[float] = None
) -> Union[Image.Image, torch.Tensor]:
    """Apply Gaussian blur.
    
    Args:
        image: Input image (PIL or tensor)
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Blurred image
    """
    if isinstance(image, Image.Image):
        validate_image(image)
        return image.filter(ImageFilter.GaussianBlur(radius=kernel_size/2))
    else:
        validate_tensor(image)
        return TF.gaussian_blur(image, kernel_size, sigma)
