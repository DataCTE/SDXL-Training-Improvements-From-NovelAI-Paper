import torch
import numpy as np
from PIL import Image

def tensor_to_pil(tensor):
    """Convert a torch tensor to PIL Image"""
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first image if batched
    tensor = tensor.cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

def pil_to_tensor(image):
    """Convert PIL Image to torch tensor"""
    np_image = np.array(image).astype(np.float32) / 255.0
    if len(np_image.shape) == 2:
        np_image = np_image[:, :, None]
    tensor = torch.from_numpy(np_image)
    tensor = tensor.permute(2, 0, 1)
    return tensor.unsqueeze(0)
