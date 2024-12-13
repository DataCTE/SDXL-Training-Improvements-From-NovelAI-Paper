import torch
import torch.nn as nn
from memory.quantization import LinearFp8
from memory.layeroffloading import quantize_layers, is_quantized_parameter, get_offload_tensor_bytes, offload_quantized


class MemoryEfficientQuantization:
    """Memory efficient quantization for SDXL training"""

    @staticmethod
    def setup_quantization(model: nn.Module, device: torch.device, train_dtype: torch.dtype):
        """Setup quantization for the model"""
        # Quantize UNet
        if hasattr(model, 'unet'):
            LinearFp8(
                model.unet,
                keep_in_fp32_modules=['time_embedding'],
                copy_parameters=True
            )
            quantize_layers(model.unet, device, train_dtype)
        
        # Quantize text encoders
        if hasattr(model, 'text_encoder_1'):
            LinearFp8(
                model.text_encoder_1,
                keep_in_fp32_modules=['embeddings'],
                copy_parameters=True
            )
            quantize_layers(model.text_encoder_1, device, train_dtype)
            
        if hasattr(model, 'text_encoder_2'):
            LinearFp8(
                model.text_encoder_2,
                keep_in_fp32_modules=['embeddings'],
                copy_parameters=True
            )
            quantize_layers(model.text_encoder_2, device, train_dtype)
        
        # Quantize VAE
        if hasattr(model, 'vae'):
            LinearFp8(
                model.vae,
                copy_parameters=True
            )
            quantize_layers(model.vae, device, train_dtype)

    @staticmethod
    def get_module_size(module: nn.Module) -> int:
        """Get memory size of a module including quantized parameters"""
        total_size = 0
        for name, param in module.named_parameters():
            if is_quantized_parameter(module, name):
                total_size += get_offload_tensor_bytes(module)
            else:
                total_size += param.numel() * param.element_size()
        return total_size

    @staticmethod
    def offload_to_cpu(module: nn.Module, non_blocking: bool = True):
        """Offload module to CPU efficiently"""
        for submodule in module.modules():
            if hasattr(submodule, 'weight'):
                offload_quantized(
                    submodule,
                    torch.device('cpu'),
                    non_blocking=non_blocking
                )