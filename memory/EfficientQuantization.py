import torch
import torch.nn as nn
from typing import Dict, Optional, List, Set
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class EfficientQuantization:
    def __init__(
        self, 
        bits: int = 8, 
        group_size: int = 128,
        use_layer_scheduling: bool = True
    ):
        self.bits = bits
        self.group_size = group_size
        self.use_layer_scheduling = use_layer_scheduling
        
        # Module tracking
        self.quantized_modules: Dict[str, nn.Module] = {}
        self.original_state: Dict[str, Dict] = {}
        
        # Layer scheduling
        self.layer_groups: Dict[str, Set[str]] = defaultdict(set)
        self.active_layers: Set[str] = set()
        self.layer_dependencies: Dict[str, List[str]] = defaultdict(list)
        
        # Memory tracking
        self.memory_per_layer: Dict[str, int] = {}
        self.total_memory_saved: int = 0
        
        # Performance optimization
        self.quantization_cache: Dict[str, Dict] = {}
        self.layer_stats: Dict[str, Dict[str, float]] = defaultdict(dict)

    def quantize_model(
        self, 
        model: nn.Module, 
        exclude_modules: Optional[set] = None,
        layer_group: Optional[str] = None
    ):
        """Quantize model with optimized memory layout and layer scheduling"""
        if exclude_modules is None:
            exclude_modules = set()
            
        for name, module in model.named_modules():
            if name in exclude_modules:
                continue
                
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Track layer group if enabled
                if layer_group and self.use_layer_scheduling:
                    self.layer_groups[layer_group].add(name)
                
                # Calculate and store memory usage
                original_memory = self._calculate_module_memory(module)
                
                # Store original state with memory optimization
                self._store_original_state(name, module)
                
                # Quantize using dynamic quantization with caching
                quantized_module = self._quantize_module(module)
                self.quantized_modules[name] = quantized_module
                
                # Update memory tracking
                quantized_memory = self._calculate_module_memory(quantized_module)
                self.memory_per_layer[name] = quantized_memory
                self.total_memory_saved += (original_memory - quantized_memory)
                
                # Track performance stats
                self._update_layer_stats(name, original_memory, quantized_memory)

    def _store_original_state(self, name: str, module: nn.Module):
        """Efficiently store original module state"""
        with torch.no_grad():
            self.original_state[name] = {
                'weight': module.weight.data.clone().cpu(),  # Store on CPU
                'bias': module.bias.data.clone().cpu() if module.bias is not None else None
            }

    def _quantize_module(self, module: nn.Module) -> nn.Module:
        """Quantize single module with caching"""
        module_key = f"{module.__class__.__name__}_{module.weight.shape}"
        
        if module_key in self.quantization_cache:
            # Use cached quantization parameters
            qconfig = self.quantization_cache[module_key]['qconfig']
        else:
            # Create new quantization config
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=(self.bits == 8)
                ),
                weight=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    ch_axis=0,
                    reduce_range=(self.bits == 8)
                )
            )
            self.quantization_cache[module_key] = {'qconfig': qconfig}

        # Quantize module
        quantized = torch.quantization.quantize_dynamic(
            module,
            {type(module)},
            dtype=torch.qint8,
            inplace=False
        )
        
        return quantized

    def activate_layer_group(self, group_name: str):
        """Activate all layers in a group"""
        if not self.use_layer_scheduling:
            return
            
        for layer_name in self.layer_groups[group_name]:
            if layer_name not in self.active_layers:
                self._activate_layer(layer_name)
        
        self.active_layers.update(self.layer_groups[group_name])

    def deactivate_layer_group(self, group_name: str):
        """Deactivate all layers in a group"""
        if not self.use_layer_scheduling:
            return
            
        for layer_name in self.layer_groups[group_name]:
            if layer_name in self.active_layers:
                self._deactivate_layer(layer_name)
                
        self.active_layers.difference_update(self.layer_groups[group_name])

    def _activate_layer(self, layer_name: str):
        """Activate single layer with dependency handling"""
        if layer_name not in self.quantized_modules:
            return
            
        # Activate dependencies first
        for dep in self.layer_dependencies[layer_name]:
            if dep not in self.active_layers:
                self._activate_layer(dep)
                
        # Restore quantized weights
        module = self.quantized_modules[layer_name]
        module.weight.data = module.weight.data.to(device='cuda')
        if module.bias is not None:
            module.bias.data = module.bias.data.to(device='cuda')

    def _deactivate_layer(self, layer_name: str):
        """Deactivate single layer"""
        if layer_name not in self.quantized_modules:
            return
            
        # Move weights to CPU
        module = self.quantized_modules[layer_name]
        module.weight.data = module.weight.data.cpu()
        if module.bias is not None:
            module.bias.data = module.bias.data.cpu()

    @staticmethod
    def _calculate_module_memory(module: nn.Module) -> int:
        """Calculate memory usage of module"""
        memory = 0
        for param in module.parameters():
            memory += param.nelement() * param.element_size()
        return memory

    def _update_layer_stats(
        self,
        name: str,
        original_memory: int,
        quantized_memory: int
    ):
        """Update performance statistics for layer"""
        self.layer_stats[name].update({
            'original_memory': original_memory,
            'quantized_memory': quantized_memory,
            'memory_saved': original_memory - quantized_memory,
            'compression_ratio': original_memory / quantized_memory
        })

    def get_memory_savings(self) -> Dict[str, float]:
        """Get memory savings statistics"""
        return {
            'total_saved_gb': self.total_memory_saved / (1024**3),
            'per_layer_saved_mb': {
                name: stats['memory_saved'] / (1024**2)
                for name, stats in self.layer_stats.items()
            },
            'compression_ratios': {
                name: stats['compression_ratio']
                for name, stats in self.layer_stats.items()
            }
        }

    def print_stats(self):
        """Print quantization statistics"""
        stats = self.get_memory_savings()
        logger.info(f"Total memory saved: {stats['total_saved_gb']:.2f} GB")
        logger.info("\nPer-layer statistics:")
        for name, saved_mb in stats['per_layer_saved_mb'].items():
            ratio = stats['compression_ratios'][name]
            logger.info(f"{name}:")
            logger.info(f"  Memory saved: {saved_mb:.2f} MB")
            logger.info(f"  Compression ratio: {ratio:.2f}x")