import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np

class LayerOffloadConductor:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.cpu_device = torch.device('cpu')
        self.streams = {
            'compute': torch.cuda.Stream(),
            'h2d': torch.cuda.Stream(),
            'd2h': torch.cuda.Stream()
        }
        self.layer_states = {}
        self.pinned_memory = {}
        
    def register_layer(self, name: str, layer: nn.Module):
        """Register a layer for offloading management"""
        self.layer_states[name] = {
            'location': 'gpu',  # Start on GPU
            'size': sum(p.numel() * p.element_size() for p in layer.parameters())
        }
        # Pre-allocate pinned memory for this layer
        self.pinned_memory[name] = {
            'gpu': None,  # Will store GPU buffer
            'cpu': torch.empty(self.layer_states[name]['size'], 
                             dtype=torch.uint8, 
                             pin_memory=True)
        }
        
    def before_layer(self, layer_idx: int, name: str):
        """Prepare layer for computation"""
        # Ensure layer is on GPU
        if self.layer_states[name]['location'] == 'cpu':
            self.load_to_gpu(name)
            
        # Wait for any pending transfers
        self.synchronize()
        
        # Switch to compute stream
        torch.cuda.current_stream().wait_stream(self.streams['h2d'])
        torch.cuda.current_stream().wait_stream(self.streams['d2h'])

    def after_layer(self, layer_idx: int, name: str):
        """Cleanup after layer computation"""
        # Ensure compute is finished
        self.streams['compute'].synchronize()
        
        # Record layer state
        self.layer_states[name]['last_used'] = layer_idx

    def offload_to_cpu(self, name: str):
        """Move layer to CPU asynchronously"""
        if self.layer_states[name]['location'] == 'cpu':
            return
            
        layer = self._get_layer_by_name(name)
        with torch.cuda.stream(self.streams['d2h']):
            # Pack parameters into contiguous buffer
            packed = self._pack_parameters(layer)
            # Copy to pre-allocated pinned memory
            self.pinned_memory[name]['gpu'] = packed
            self.pinned_memory[name]['cpu'].copy_(packed, non_blocking=True)
            
        # Update state
        self.layer_states[name]['location'] = 'cpu'
        
        # Move layer parameters to CPU
        for param in layer.parameters():
            param.data = param.data.to(self.cpu_device)
            
    def load_to_gpu(self, name: str):
        """Move layer to GPU asynchronously"""
        if self.layer_states[name]['location'] == 'gpu':
            return
            
        layer = self._get_layer_by_name(name)
        with torch.cuda.stream(self.streams['h2d']):
            # Copy from pinned memory to GPU
            gpu_buffer = self.pinned_memory[name]['cpu'].to(
                self.device, non_blocking=True)
            # Unpack parameters
            self._unpack_parameters(layer, gpu_buffer)
            
        # Update state
        self.layer_states[name]['location'] = 'gpu'
        
    def _get_layer_by_name(self, name: str) -> nn.Module:
        """Get layer by name from model"""
        return dict(self.model.named_modules())[name]
        
    def _pack_parameters(self, layer: nn.Module) -> torch.Tensor:
        """Pack layer parameters into contiguous buffer"""
        return torch.cat([p.data.view(-1) for p in layer.parameters()])
        
    def _unpack_parameters(self, layer: nn.Module, buffer: torch.Tensor):
        """Unpack buffer into layer parameters"""
        offset = 0
        for param in layer.parameters():
            numel = param.numel()
            param.data = buffer[offset:offset + numel].view(param.shape)
            offset += numel
            
    def synchronize(self):
        """Synchronize all CUDA streams"""
        for stream in self.streams.values():
            stream.synchronize()

def ensure_three_channels(x):
    return x[:3]

def convert_to_bfloat16(x):
    return x.to(torch.bfloat16)

class LayerOffloadStrategy:
    def __init__(
        self,
        model: nn.Module,
        max_vram_usage: float = 0.8
    ):
        """
        Initialize layer offload strategy
        
        Args:
            model: The model to manage memory for
            max_vram_usage: Maximum fraction of total VRAM to use (0.0-1.0)
        """
        self.model = model
        self.max_vram = int(torch.cuda.get_device_properties(0).total_memory * max_vram_usage)
        self.layer_sizes = self._calculate_layer_sizes()
        self.layer_dependencies = self._analyze_dependencies()
        self.current_vram_usage = 0
        
    def _calculate_layer_sizes(self) -> Dict[str, int]:
        """Calculate memory size of each layer"""
        sizes = {}
        for name, module in self.model.named_modules():
            if len(list(module.parameters())) > 0:  # Only count parameterized layers
                sizes[name] = sum(p.numel() * p.element_size() for p in module.parameters())
        return sizes
        
    def _analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analyze layer dependencies for forward/backward passes"""
        deps = {}
        # For SDXL UNet, analyze dependencies based on layer names
        for name in self.layer_sizes.keys():
            deps[name] = []
            parts = name.split('.')
            
            # Handle different layer types
            if any(x in name for x in ['down_blocks', 'up_blocks', 'mid_block']):
                # For down blocks
                if 'down_blocks' in name:
                    try:
                        block_num = int([p for p in parts if p.isdigit()][0])
                        # Add dependencies on previous down blocks
                        for i in range(block_num):
                            deps[name].extend([
                                k for k in self.layer_sizes.keys() 
                                if f'down_blocks.{i}.' in k
                            ])
                    except (IndexError, ValueError):
                        continue
                        
                # For middle block
                elif 'mid_block' in name:
                    # Middle block depends on all down blocks
                    deps[name].extend([
                        k for k in self.layer_sizes.keys()
                        if 'down_blocks' in k
                    ])
                    
                # For up blocks
                elif 'up_blocks' in name:
                    try:
                        block_num = int([p for p in parts if p.isdigit()][0])
                        # Up blocks depend on middle block and corresponding down block
                        deps[name].extend([
                            k for k in self.layer_sizes.keys()
                            if 'mid_block' in k or f'down_blocks.{3-block_num}.' in k
                        ])
                    except (IndexError, ValueError):
                        continue
                        
            # Remove self-dependencies
            deps[name] = [d for d in deps[name] if d != name]
            
        return deps
        
    def get_required_layers(self, current_layer: str, phase: str = 'forward') -> List[str]:
        """Get layers required to be in VRAM for current computation"""
        required = set([current_layer])
        
        if phase == 'forward':
            # Add dependencies for forward pass
            required.update(self.layer_dependencies[current_layer])
        else:  # backward
            # For backward pass, we need:
            # 1. Current layer for gradient computation
            # 2. Dependent layers that will need this layer's gradient
            for name, deps in self.layer_dependencies.items():
                if current_layer in deps:
                    required.add(name)
                    
        return list(required)
        
    def update_vram_usage(self, layer_name: str, in_vram: bool):
        """Update tracked VRAM usage when moving layers"""
        if layer_name in self.layer_sizes:
            size = self.layer_sizes[layer_name]
            self.current_vram_usage += size if in_vram else -size
            
    def can_fit_in_vram(self, layer_names: List[str]) -> bool:
        """Check if given layers can fit in VRAM"""
        required_memory = sum(self.layer_sizes[name] for name in layer_names)
        return (self.current_vram_usage + required_memory) <= self.max_vram
        
    def suggest_offload(self, required_layers: List[str]) -> List[str]:
        """Suggest layers to offload to make room for required layers"""
        if self.can_fit_in_vram(required_layers):
            return []
            
        # Calculate how much memory we need to free
        required_memory = sum(self.layer_sizes[name] for name in required_layers)
        memory_to_free = (self.current_vram_usage + required_memory) - self.max_vram
        
        # Find layers to offload, prioritizing:
        # 1. Layers not in required_layers
        # 2. Largest layers first to minimize transfers
        candidates = sorted(
            [name for name in self.layer_sizes.keys() if name not in required_layers],
            key=lambda x: self.layer_sizes[x],
            reverse=True
        )
        
        to_offload = []
        freed_memory = 0
        for name in candidates:
            if freed_memory >= memory_to_free:
                break
            to_offload.append(name)
            freed_memory += self.layer_sizes[name]
            
        return to_offload

class StaticLayerAllocator:
    def __init__(self, total_size: int, device: torch.device):
        """
        Initialize static memory allocator for layer parameters
        
        Args:
            total_size: Total size in bytes to allocate
            device: Device to allocate memory on
        """
        self.device = device
        # Allocate one large contiguous buffer
        self.memory = torch.empty(total_size, dtype=torch.uint8, device=device)
        self.allocated_regions = {}  # name -> (start, size)
        self.free_regions = [(0, total_size)]  # List of (start, size)
        
    def allocate(self, name: str, size: int) -> torch.Tensor:
        """Allocate memory region of given size"""
        if name in self.allocated_regions:
            start, allocated_size = self.allocated_regions[name]
            if allocated_size >= size:
                return self._get_region(start, size)
            else:
                # Need to free and reallocate larger region
                self.free(name)
                
        # Find free region
        best_fit = None
        best_idx = None
        for idx, (start, free_size) in enumerate(self.free_regions):
            if free_size >= size:
                if best_fit is None or free_size < best_fit[1]:
                    best_fit = (start, free_size)
                    best_idx = idx
                    
        if best_fit is None:
            raise RuntimeError(f"Failed to allocate {size} bytes")
            
        start, free_size = best_fit
        self.free_regions.pop(best_idx)
        
        # If there's remaining space, add new free region
        if free_size > size:
            self.free_regions.append((start + size, free_size - size))
            
        self.allocated_regions[name] = (start, size)
        return self._get_region(start, size)
        
    def free(self, name: str):
        """Free allocated region"""
        if name not in self.allocated_regions:
            return
            
        start, size = self.allocated_regions.pop(name)
        
        # Merge with adjacent free regions
        merged = False
        for idx, (free_start, free_size) in enumerate(self.free_regions):
            if free_start + free_size == start:  # Merge with region before
                self.free_regions[idx] = (free_start, free_size + size)
                merged = True
                break
            elif start + size == free_start:  # Merge with region after
                self.free_regions[idx] = (start, free_size + size)
                merged = True
                break
                
        if not merged:
            self.free_regions.append((start, size))
            
        # Sort free regions for better allocation
        self.free_regions.sort(key=lambda x: x[0])
        
    def _get_region(self, start: int, size: int) -> torch.Tensor:
        """Get view of memory region"""
        return self.memory[start:start + size]

class StaticActivationAllocator:
    def __init__(self, model: nn.Module):
        """
        Initialize activation memory manager
        
        Args:
            model: The model to manage activations for
        """
        self.model = model
        self.activation_sizes = {}  # layer_name -> activation_size
        self.stored_activations = {}  # layer_name -> activation_tensor
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to track activation sizes"""
        def hook_fn(name, module, input, output):
            # Store activation size if larger than previous
            if isinstance(output, torch.Tensor):
                size = output.numel() * output.element_size()
            elif isinstance(output, (tuple, list)):
                size = sum(t.numel() * t.element_size() for t in output 
                          if isinstance(t, torch.Tensor))
            else:
                return
                
            if name not in self.activation_sizes or size > self.activation_sizes[name]:
                self.activation_sizes[name] = size
                
        for name, module in self.model.named_modules():
            if len(list(module.parameters())) > 0:  # Only track parameterized layers
                hook = module.register_forward_hook(
                    lambda mod, inp, out, n=name: hook_fn(n, mod, inp, out)
                )
                self.hooks.append(hook)
                
    def allocate_buffers(self, device: torch.device):
        """Allocate buffers for storing activations"""
        total_size = sum(self.activation_sizes.values())
        self.memory = torch.empty(total_size, dtype=torch.uint8, device=device)
        
        # Allocate regions for each layer
        offset = 0
        for name, size in self.activation_sizes.items():
            self.stored_activations[name] = self.memory[offset:offset + size]
            offset += size
            
    def store_activation(self, name: str, activation: torch.Tensor):
        """Store activation in pre-allocated buffer"""
        if name not in self.stored_activations:
            return
            
        buffer = self.stored_activations[name]
        if activation.numel() * activation.element_size() > buffer.numel():
            return  # Buffer too small, skip storing
            
        # Copy activation to buffer
        flat_activation = activation.view(-1)
        buffer[:flat_activation.numel()].copy_(
            flat_activation.view(torch.uint8), non_blocking=True
        )
        
    def retrieve_activation(self, name: str, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """Retrieve stored activation"""
        if name not in self.stored_activations:
            raise KeyError(f"No stored activation for {name}")
            
        buffer = self.stored_activations[name]
        size = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
        
        if size > buffer.numel():
            raise RuntimeError(f"Buffer too small for activation {name}")
            
        # Copy from buffer to new tensor
        activation = torch.empty(shape, dtype=dtype, device=buffer.device)
        buffer[:size].view(dtype)[:activation.numel()].copy_(
            activation.view(-1), non_blocking=True
        )
        return activation
        
    def clear(self):
        """Clear all stored activations"""
        self.stored_activations.clear()
        
    def __del__(self):
        """Remove hooks when deallocated"""
        for hook in self.hooks:
            hook.remove()

