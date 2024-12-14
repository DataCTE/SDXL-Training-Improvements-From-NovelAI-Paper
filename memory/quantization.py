try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear8bitLt, LinearNf4
    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False
    Linear8bitLt = None
    LinearNf4 = None
    bnb = None

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Any
import logging
import math
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for quantization settings"""
    bits: int = 8  # Number of bits for quantization
    group_size: int = 128  # Size of groups for quantization
    symmetric: bool = True  # Whether to use symmetric quantization
    calibration_steps: int = 100  # Number of steps for calibration
    dynamic_method: str = 'percentile'  # Method for dynamic quantization
    outlier_threshold: float = 3.0  # Threshold for outlier detection
    use_cache: bool = True  # Whether to cache quantization parameters

class QuantizedModuleMixin(metaclass=ABCMeta):
    """Enhanced base mixin for quantized modules"""
    @abstractmethod 
    def quantize(self, device: Optional[torch.device] = None) -> None:
        pass
        
    @abstractmethod
    def dequantize(self) -> None:
        pass
        
    @abstractmethod
    def calibrate(self, data: torch.Tensor) -> None:
        pass

class QuantizedLinearMixin(metaclass=ABCMeta):
    """Enhanced base mixin for quantized linear layers"""
    @abstractmethod
    def original_weight_shape(self) -> Tuple[int, ...]:
        pass

    @abstractmethod 
    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pass
        
    @abstractmethod
    def compress(self) -> None:
        """Compress weights for storage"""
        pass
        
    @abstractmethod
    def decompress(self) -> None:
        """Decompress weights for computation"""
        pass

class LinearFp8(nn.Linear, QuantizedModuleMixin, QuantizedLinearMixin):
    """Enhanced 8-bit floating point quantized linear layer"""
    def __init__(
        self, 
        *args, 
        config: Optional[QuantizationConfig] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = config or QuantizationConfig()
        self.is_quantized = False
        self.fp8_dtype = torch.float8_e4m3fn
        self._scale = torch.tensor(1.0, dtype=torch.float)
        self.register_buffer("scale", self._scale)
        self.compute_dtype = None
        
        # Caching for quantization parameters
        self._quantization_cache: Dict[str, Any] = {}
        self._outliers: Optional[torch.Tensor] = None
        
        # Initialize with improved scaling
        self._init_parameters()
        
    def _init_parameters(self) -> None:
        """Initialize parameters with improved scaling"""
        if self.weight is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.weight, -bound, bound)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
                
    def original_weight_shape(self) -> Tuple[int, ...]:
        return self.weight.shape

    def unquantized_weight(
        self, 
        dtype: torch.dtype, 
        device: torch.device
    ) -> torch.Tensor:
        """Get unquantized weight with caching"""
        cache_key = f"{dtype}_{device}"
        if cache_key not in self._quantization_cache:
            if self._scale is not None:
                weight = self.weight.detach().to(dtype) * self._scale.to(dtype=dtype)
            else:
                weight = self.weight.detach().to(dtype=dtype)
            self._quantization_cache[cache_key] = weight
        return self._quantization_cache[cache_key]

    def quantize(self, device: Optional[torch.device] = None) -> None:
        """Enhanced quantization with outlier handling and error recovery"""
        if self.is_quantized:
            return
            
        try:
            weight = self.weight.data
            orig_device = weight.device
            
            if device is not None:
                weight = weight.to(device=device)
                
            # Handle outliers
            if self.config.outlier_threshold > 0:
                weight, outliers = self._handle_outliers(weight)
                self._outliers = outliers
                
            # Compute quantization parameters
            if self.config.dynamic_method == 'percentile':
                abs_max = torch.quantile(weight.abs(), 0.99999)
            else:
                abs_max = weight.abs().max()
                
            # Apply scaling with stability checks
            scale = torch.clamp(abs_max, min=1e-12) / torch.finfo(self.fp8_dtype).max
            self._scale.copy_(scale)
            
            # Quantize weights
            weight = weight.div_(scale).to(dtype=self.fp8_dtype)
            
            if device is not None:
                weight = weight.to(device=orig_device)
                
            self.weight.data = weight
            self.is_quantized = True
            
            # Clear cache
            self._quantization_cache.clear()
            
        except Exception as e:
            logger.error(f"Error during quantization: {e}")
            self._recover_from_failed_quantization()
            raise
            
    def _handle_outliers(
        self, 
        weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle outliers in weight distribution"""
        std = weight.std()
        threshold = std * self.config.outlier_threshold
        outliers = (weight.abs() > threshold).float()
        
        if outliers.sum() > 0:
            # Store outliers separately
            outlier_values = weight * outliers
            weight = weight * (1 - outliers)
            return weight, outlier_values
        return weight, None
        
    def _recover_from_failed_quantization(self) -> None:
        """Recover from failed quantization"""
        self.is_quantized = False
        if hasattr(self, '_orig_weight'):
            self.weight.data = self._orig_weight
        self._scale.fill_(1.0)
        self._quantization_cache.clear()
        
    def dequantize(self) -> None:
        """Dequantize weights"""
        if not self.is_quantized:
            return
            
        try:
            weight = self.weight.data
            weight = weight.to(torch.float32) * self._scale
            
            # Restore outliers if present
            if self._outliers is not None:
                weight = weight + self._outliers
                
            self.weight.data = weight
            self.is_quantized = False
            self._quantization_cache.clear()
            
        except Exception as e:
            logger.error(f"Error during dequantization: {e}")
            raise
            
    def calibrate(self, data: torch.Tensor) -> None:
        """Calibrate quantization parameters using input data"""
        if not self.training:
            return
            
        try:
            with torch.no_grad():
                # Compute activation statistics
                act_max = data.abs().max().item()
                act_std = data.std().item()
                
                # Update scaling factor based on activations
                weight_scale = self._scale.item()
                combined_scale = math.sqrt(weight_scale * act_max)
                self._scale.fill_(combined_scale)
                
                # Update outlier threshold
                if self.config.outlier_threshold > 0:
                    self.config.outlier_threshold = max(3.0, act_std * 2)
                    
        except Exception as e:
            logger.warning(f"Error during calibration: {e}")
            
    def compress(self) -> None:
        """Compress weights for storage"""
        if not self.is_quantized:
            return
            
        try:
            # Store original weight and compress
            self._orig_weight = self.weight.data.clone()
            compressed = self._compress_fp8(self.weight.data)
            self.weight.data = compressed
            
        except Exception as e:
            logger.error(f"Error during compression: {e}")
            raise
            
    def decompress(self) -> None:
        """Decompress weights for computation"""
        if not hasattr(self, '_orig_weight'):
            return
            
        try:
            self.weight.data = self._orig_weight
            delattr(self, '_orig_weight')
            
        except Exception as e:
            logger.error(f"Error during decompression: {e}")
            raise
            
    def _compress_fp8(self, weight: torch.Tensor) -> torch.Tensor:
        """Compress FP8 tensor efficiently"""
        bits = torch.zeros_like(weight, dtype=torch.uint8)
        bits.copy_((weight.view(torch.uint8) & 0xFF))
        return bits
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with dynamic type handling"""
        try:
            weight = self.weight.detach()
            target_dtype = self.compute_dtype if self.compute_dtype is not None else x.dtype
            
            # Use cached conversion if available
            weight = self.unquantized_weight(target_dtype, x.device)
            
            # Compute output with error checking
            try:
                out = F.linear(x, weight, self.bias)
            except RuntimeError as e:
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    logger.error(f"Shape mismatch in linear layer: x={x.shape}, weight={weight.shape}")
                raise
                
            return out
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise

def replace_linear_with_fp8_layers(
    parent_module: nn.Module,
    keep_in_fp32_modules: Optional[List[str]] = None,
    copy_parameters: bool = False,
    config: Optional[QuantizationConfig] = None
) -> None:
    """Replace linear layers with enhanced FP8 quantized versions"""
    if keep_in_fp32_modules is None:
        keep_in_fp32_modules = []
        
    visited = set()
    
    try:
        def _replace_recursive(module: nn.Module, prefix: str = "") -> None:
            if id(module) in visited:
                return
            visited.add(id(module))
            
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if full_name in keep_in_fp32_modules:
                    continue
                    
                if isinstance(child, nn.Linear):
                    # Create FP8 layer with configuration
                    fp8_linear = LinearFp8(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        config=config
                    )
                    
                    # Copy parameters if requested
                    if copy_parameters:
                        with torch.no_grad():
                            fp8_linear.weight.data = child.weight.data
                            if child.bias is not None:
                                fp8_linear.bias.data = child.bias.data
                                
                    # Replace module
                    setattr(module, name, fp8_linear)
                    logger.debug(f"Replaced {full_name} with FP8 layer")
                    
                else:
                    _replace_recursive(child, full_name)
                    
        _replace_recursive(parent_module)
        
    except Exception as e:
        logger.error(f"Error replacing linear layers: {e}")
        raise

def is_quantized_parameter(module: nn.Module, name: str) -> bool:
    """Enhanced check for quantized parameters"""
    if isinstance(module, LinearFp8):
        return name == "weight"
    if HAVE_BNB:
        if isinstance(module, LinearNf4):
            return name in ["weight", "absmax", "offset", "code"]
        if isinstance(module, Linear8bitLt):
            return name == "weight"
    return False

def get_offload_tensors(module: nn.Module) -> List[torch.Tensor]:
    """Get tensors that should be offloaded with improved handling"""
    tensors = []
    try:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight') and module.weight is not None:
                tensors.append(module.weight)
            if getattr(module, 'bias', None) is not None:
                tensors.append(module.bias)
                
        if HAVE_BNB and isinstance(module, LinearNf4):
            if hasattr(module.quant_state, 'absmax'):
                tensors.append(module.quant_state.absmax)
                
    except Exception as e:
        logger.error(f"Error getting offload tensors: {e}")
        
    return tensors

def get_offload_tensor_bytes(module: nn.Module) -> int:
    """Get total bytes needed for offloading with error handling"""
    try:
        tensors = get_offload_tensors(module)
        return sum(t.element_size() * t.numel() for t in tensors)
    except Exception as e:
        logger.error(f"Error calculating offload bytes: {e}")
        return 0

def offload_quantized(
    module: nn.Module,
    device: torch.device,
    non_blocking: bool = False,
    allocator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
) -> None:
    """Enhanced offload of quantized tensors with error handling"""
    try:
        tensors = get_offload_tensors(module)
        
        if allocator is None:
            # Basic offload
            for tensor in tensors:
                tensor.data = tensor.data.to(
                    device=device,
                    non_blocking=non_blocking
                )
        else:
            # Custom allocation
            for tensor in tensors:
                try:
                    new_tensor = allocator(tensor)
                    new_tensor.copy_(
                        tensor.data,
                        non_blocking=non_blocking
                    )
                    tensor.data = new_tensor
                except Exception as e:
                    logger.error(f"Error in custom allocation: {e}")
                    # Fallback to basic offload
                    tensor.data = tensor.data.to(
                        device=device,
                        non_blocking=non_blocking
                    )
                    
    except Exception as e:
        logger.error(f"Error offloading quantized tensors: {e}")
        raise

class EfficientQuantization:
    def __init__(self):
        self.quantized_modules = {}
        
    def quantize_model(self, model: nn.Module):
        """Apply efficient 8-bit quantization"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.quantized_modules[name] = torch.ao.quantization.quantize_dynamic(
                    module,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )