try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear8bitLt, LinearNf4  # Add LinearNf4
    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False
    Linear8bitLt = None
    LinearNf4 = None  # Add this
    bnb = None

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizedModuleMixin(metaclass=ABCMeta):
    """Base mixin for quantized modules"""
    @abstractmethod 
    def quantize(self, device: torch.device | None = None):
        pass

class QuantizedLinearMixin(metaclass=ABCMeta):
    """Base mixin for quantized linear layers"""
    @abstractmethod
    def original_weight_shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod 
    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pass

class LinearFp8(nn.Linear, QuantizedModuleMixin, QuantizedLinearMixin):
    """8-bit floating point quantized linear layer"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_quantized = False
        self.fp8_dtype = torch.float8_e4m3fn
        self._scale = torch.tensor(1.0, dtype=torch.float)
        self.register_buffer("scale", self._scale)
        self.compute_dtype = None

    def original_weight_shape(self) -> tuple[int, ...]:
        return self.weight.shape

    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self._scale is not None:
            return self.weight.detach().to(dtype) * self._scale.to(dtype=dtype)
        return self.weight.detach().to(dtype=dtype)

    def quantize(self, device: torch.device | None = None):
        if self.is_quantized:
            return
        self.is_quantized = True

        weight = self.weight.data
        orig_device = weight.device
        if weight.dtype != self.fp8_dtype:
            if device is not None:
                weight = weight.to(device=device)

            abs_max = weight.abs().max()
            self._scale.copy_(torch.clamp(abs_max, min=1e-12) / torch.finfo(self.fp8_dtype).max)
            weight = weight.div_(self._scale).to(dtype=self.fp8_dtype)

            if device is not None:
                weight = weight.to(device=orig_device)
        self.weight.data = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.detach()
        weight = weight.to(dtype=self.compute_dtype if self.compute_dtype is not None else x.dtype)

        if self._scale is not None:
            weight = weight.mul_(self._scale)
        x = F.linear(x, weight, self.bias)
        return x

def replace_linear_with_fp8_layers(
        parent_module: nn.Module,
        keep_in_fp32_modules: list[str] | None = None,
        copy_parameters: bool = False,
):
    """Replace linear layers with FP8 quantized versions"""
    if keep_in_fp32_modules is None:
        keep_in_fp32_modules = []

    visited = set()

    def _replace_recursive(module: nn.Module, prefix: str = ""):
        if id(module) in visited:
            return
        visited.add(id(module))

        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if full_name in keep_in_fp32_modules:
                continue

            if isinstance(child, nn.Linear):
                fp8_linear = LinearFp8(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None
                )
                if copy_parameters:
                    fp8_linear.weight.data = child.weight.data
                    if child.bias is not None:
                        fp8_linear.bias.data = child.bias.data
                setattr(module, name, fp8_linear)
            else:
                _replace_recursive(child, full_name)

    _replace_recursive(parent_module)

def is_quantized_parameter(module: nn.Module, name: str) -> bool:
    """Check if parameter should be quantized"""
    if isinstance(module, LinearFp8):
        return name == "weight"
    if HAVE_BNB:
        if isinstance(module, LinearNf4):
            return name in ["weight", "absmax", "offset", "code"]
        if isinstance(module, Linear8bitLt):
            return name == "weight"
    return False

def get_offload_tensors(module: nn.Module) -> list[torch.Tensor]:
    """Get tensors that should be offloaded"""
    tensors = []
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        tensors.append(module.weight)
        if getattr(module, 'bias', None) is not None:
            tensors.append(module.bias)
    if HAVE_BNB and isinstance(module, LinearNf4):
        tensors.append(module.quant_state.absmax)
    return tensors

def get_offload_tensor_bytes(module: nn.Module) -> int:
    """Get total bytes needed for offloading"""
    tensors = get_offload_tensors(module)
    return sum(t.element_size() * t.numel() for t in tensors)

def offload_quantized(
    module: nn.Module,
    device: torch.device,
    non_blocking: bool = False,
    allocator: Callable[[torch.Tensor], torch.Tensor] | None = None
):
    """Offload quantized tensors to specified device"""
    tensors = get_offload_tensors(module)
    
    if allocator is None:
        for tensor in tensors:
            tensor.data = tensor.data.to(device=device, non_blocking=non_blocking)
    else:
        for tensor in tensors:
            new_tensor = allocator(tensor)
            new_tensor.copy_(tensor.data, non_blocking=non_blocking)
            tensor.data = new_tensor