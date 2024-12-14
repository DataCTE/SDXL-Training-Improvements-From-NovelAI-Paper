import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np

class LayerOffloadConductor:
    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.layers = {}
        self.pinned_memory = {}
        self.current_layer = None
        self.layer_states = {}  # Track layer states (gpu/cpu)
        self.total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Optimize stream usage with dedicated streams
        self.streams = {
            'h2d': torch.cuda.Stream(),  # Host to Device
            'd2h': torch.cuda.Stream(),  # Device to Host
            'compute': torch.cuda.Stream()  # Computation
        }
        
        # Pre-allocate reusable buffers for transfers
        self.transfer_buffers = {
            'h2d': None,
            'd2h': None
        }
        
        # Cache for parameter metadata
        self.param_cache = {}
        
        # Enable tensor cores if available
        if torch.cuda.get_device_capability(device)[0] >= 7:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
    def register_layer(self, name: str, layer: nn.Module):
        """Register a layer for offloading management with improved error handling and memory management"""
        try:
            if name in self.layers:
                raise ValueError(f"Layer {name} already registered")
                
            self.layers[name] = layer
            self.layer_states[name] = 'gpu'  # Start on GPU
            
            # Pre-calculate and cache parameter metadata
            total_params = 0
            param_info = []
            
            # Get all parameters that can be pinned
            pinnable_params = []
            for p in layer.parameters():
                if not isinstance(p, torch.Tensor) or p.numel() <= 0:
                    continue
                    
                # Only try to pin contiguous tensors
                if p.is_contiguous():
                    pinnable_params.append(p)
                
                param_info.append({
                    'numel': p.numel(),
                    'shape': p.shape,
                    'dtype': p.dtype,
                    'id': id(p),
                    'requires_grad': p.requires_grad,
                    'is_pinnable': p.is_contiguous()
                })
                total_params += p.numel()
            
            if total_params > 0:
                try:
                    # Calculate buffer size based on pinnable parameters only
                    pinnable_size = sum(p.numel() * p.element_size() for p in pinnable_params)
                    buffer_size = min(
                        pinnable_size * 2,  # Double buffer
                        self.total_gpu_memory // 8  # Limit to 1/8th of GPU memory
                    )
                    
                    # Only allocate pinned memory if we have pinnable parameters
                    if pinnable_params:
                        try:
                            pinned_buffer = torch.empty(
                                buffer_size, 
                                dtype=torch.uint8,  # Use uint8 for more efficient memory usage
                                pin_memory=True,
                                device='cpu'
                            )
                            is_pinned = True
                        except RuntimeError:
                            # Fallback to non-pinned memory
                            pinned_buffer = torch.empty(
                                buffer_size,
                                dtype=torch.uint8,
                                device='cpu'
                            )
                            is_pinned = False
                            print(f"Warning: Failed to allocate pinned memory for {name}, falling back to regular memory")
                    else:
                        # Create regular buffer for non-pinnable parameters
                        pinned_buffer = torch.empty(
                            buffer_size,
                            dtype=torch.uint8,
                            device='cpu'
                        )
                        is_pinned = False
                        
                    self.pinned_memory[name] = {
                        'cpu': pinned_buffer,
                        'size': buffer_size,
                        'total_params': total_params,
                        'param_info': param_info,
                        'pinned': is_pinned,
                        'pinnable_params': [id(p) for p in pinnable_params]
                    }
                    
                except RuntimeError as e:
                    print(f"Warning: Failed to allocate memory for {name}: {e}")
                    # Fallback with minimal allocation
                    buffer_size = min(total_params * 4, self.total_gpu_memory // 16)
                    self.pinned_memory[name] = {
                        'cpu': torch.empty(buffer_size, dtype=torch.uint8, device='cpu'),
                        'size': buffer_size,
                        'total_params': total_params,
                        'param_info': param_info,
                        'pinned': False,
                        'pinnable_params': []
                    }
            
            # Cache parameter access patterns
            self.param_cache[name] = {
                'param_list': list(layer.parameters()),
                'total_size': total_params,
                'access_count': 0
            }
            
        except Exception as e:
            print(f"Error registering layer {name}: {e}")
            raise
    
    def before_layer(self, layer_idx: int, name: str):
        """Prepare layer for computation with safety checks"""
        if name not in self.layers:
            return
            
        try:
            self.current_layer = name
            
            # Skip if already on GPU
            if self.layer_states.get(name) == 'gpu':
                return
                
            # Optimize transfer with event synchronization
            current = torch.cuda.current_stream()
            with torch.cuda.stream(self.streams['h2d']):
                # Wait for any pending operations
                current.synchronize()
                self.move_to_gpu(name)
                
            # Record completion event
            event = torch.cuda.Event()
            self.streams['h2d'].record_event(event)
            current.wait_event(event)
            
        except Exception as e:
            print(f"Error in before_layer for {name}: {e}")
            self.handle_error(name)
    
    def after_layer(self, layer_idx: int, name: str):
        """Cleanup after layer computation with error recovery"""
        if name not in self.layers:
            return
            
        try:
            # Optimize stream synchronization
            compute_event = torch.cuda.Event()
            torch.cuda.current_stream().record_event(compute_event)
            
            with torch.cuda.stream(self.streams['d2h']):
                compute_event.wait()
                self.offload_to_cpu(name)
                
            self.current_layer = None
            
        except Exception as e:
            print(f"Error in after_layer for {name}: {e}")
            self.handle_error(name)
    
    def move_to_gpu(self, name: str):
        """Move layer parameters to GPU with optimized transfers"""
        if name not in self.layers or name not in self.pinned_memory:
            return
            
        try:
            layer = self.layers[name]
            mem_info = self.pinned_memory[name]
            cached_params = self.param_cache[name]['param_list']
            
            # Check memory with headroom
            if torch.cuda.memory_allocated() + mem_info['total_params'] * 2.2 > self.total_gpu_memory * 0.95:
                self.handle_oom(name)
                return
            
            # Optimize bulk transfer
            offset = 0
            transfer_buffer = self.transfer_buffers['h2d']
            
            for param_info in mem_info['param_info']:
                numel = param_info['numel']
                if numel == 0:
                    continue
                
                # Ensure buffer capacity
                if offset + numel > mem_info['size']:
                    try:
                        new_size = min(max(offset + numel, mem_info['size'] * 2),
                                     self.total_gpu_memory // 4)
                        mem_info['cpu'] = torch.empty(new_size, 
                                                    dtype=torch.float16,
                                                    pin_memory=mem_info.get('pinned', True),
                                                    device='cpu')
                        mem_info['size'] = new_size
                        
                        # Update transfer buffer if needed
                        if transfer_buffer.numel() < new_size:
                            self.transfer_buffers['h2d'] = torch.empty(
                                new_size,
                                dtype=torch.float16,
                                device=self.device,
                                pin_memory=True
                            )
                            transfer_buffer = self.transfer_buffers['h2d']
                            
                    except RuntimeError as e:
                        print(f"Warning: Failed to resize buffer for {name}: {e}")
                        return
                
                try:
                    # Optimized copy with shape pre-validation
                    param = cached_params[offset // numel]
                    if param.shape == param_info['shape']:
                        # Use non-blocking transfer with proper stream
                        param.data.copy_(
                            mem_info['cpu'][offset:offset + numel].view_as(param),
                            non_blocking=True
                        )
                except RuntimeError as e:
                    print(f"Error copying parameter to GPU for {name}: {e}")
                    self.handle_error(name)
                    return
                
                offset += numel
            
            self.layer_states[name] = 'gpu'
            
        except Exception as e:
            print(f"Error in move_to_gpu for {name}: {e}")
            self.handle_error(name)
    
    def offload_to_cpu(self, name: str):
        """Offload layer parameters to CPU with optimized transfers"""
        if name not in self.layers or name not in self.pinned_memory:
            return
            
        try:
            layer = self.layers[name]
            mem_info = self.pinned_memory[name]
            cached_params = self.param_cache[name]['param_list']
            
            # Optimize bulk transfer
            offset = 0
            transfer_buffer = self.transfer_buffers['d2h']
            
            for param_info in mem_info['param_info']:
                numel = param_info['numel']
                if numel == 0:
                    continue
                
                # Ensure buffer capacity
                if offset + numel > mem_info['size']:
                    try:
                        new_size = min(max(offset + numel, mem_info['size'] * 2),
                                     self.total_gpu_memory // 4)
                        mem_info['cpu'] = torch.empty(new_size, 
                                                    dtype=torch.float16,
                                                    pin_memory=mem_info.get('pinned', True),
                                                    device='cpu')
                        mem_info['size'] = new_size
                        
                        # Update transfer buffer if needed
                        if transfer_buffer.numel() < new_size:
                            self.transfer_buffers['d2h'] = torch.empty(
                                new_size,
                                dtype=torch.float16,
                                device='cpu',
                                pin_memory=True
                            )
                            transfer_buffer = self.transfer_buffers['d2h']
                            
                    except RuntimeError as e:
                        print(f"Warning: Failed to resize buffer for {name}: {e}")
                        return
                
                try:
                    param = cached_params[offset // numel]
                    # Optimized copy with pre-conversion
                    if param.is_cuda:  # Only copy if still on GPU
                        mem_info['cpu'][offset:offset + numel].copy_(
                            param.data.to(torch.float16).view(-1),
                            non_blocking=True
                        )
                        param.data = param.data.cpu()  # Move to CPU after copy
                        
                except RuntimeError as e:
                    print(f"Error copying parameter to CPU for {name}: {e}")
                    self.handle_error(name)
                    return
                
                offset += numel
            
            self.layer_states[name] = 'cpu'
            
        except Exception as e:
            print(f"Error in offload_to_cpu for {name}: {e}")
            self.handle_error(name)
            
    def handle_error(self, name: str):
        """Handle errors during layer operations"""
        try:
            torch.cuda.empty_cache()
            self.layer_states[name] = 'unknown'
            
            # Fast emergency CPU offload
            if name in self.layers:
                layer = self.layers[name]
                for p in layer.parameters():
                    if isinstance(p, torch.Tensor) and p.is_cuda:
                        try:
                            p.data = p.data.cpu()
                        except:
                            pass
                            
        except Exception as e:
            print(f"Error in error handler for {name}: {e}")
            
    def handle_oom(self, name: str):
        """Handle out-of-memory situations with optimized cleanup"""
        try:
            torch.cuda.empty_cache()
            
            # Prioritized memory recovery
            gpu_layers = [(n, self.pinned_memory[n]['total_params']) 
                         for n, state in self.layer_states.items()
                         if state == 'gpu' and n != name]
            
            # Sort by size for optimal recovery
            for other_name, _ in sorted(gpu_layers, key=lambda x: x[1], reverse=True):
                try:
                    self.offload_to_cpu(other_name)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error in OOM handler: {e}")
            
    def clear(self):
        """Optimized cleanup of all resources"""
        try:
            # Fast synchronization
            current = torch.cuda.current_stream()
            for stream in self.streams.values():
                if stream != current:
                    stream.synchronize()
            
            # Bulk cleanup
            for name in self.layers:
                if self.layer_states.get(name) == 'gpu':
                    try:
                        self.offload_to_cpu(name)
                    except:
                        pass
            
            # Clear memory
            self.pinned_memory.clear()
            self.transfer_buffers = {'h2d': None, 'd2h': None}
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error in clear: {e}")
            
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.clear()
        except:
            pass

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

