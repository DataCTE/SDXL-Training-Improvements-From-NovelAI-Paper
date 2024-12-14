import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass
import logging
import warnings
from collections import defaultdict
from enum import Enum, auto
import time

class LayerOffloadStrategy(Enum):
    """Enum for different layer offloading strategies"""
    STATIC = auto()  # Keep layer allocation static
    DYNAMIC = auto()  # Dynamically move layers between CPU/GPU
    ADAPTIVE = auto()  # Adaptively adjust based on memory pressure

@dataclass
class LayerMemoryInfo:
    """Dataclass to store layer memory information"""
    size: int
    pinned: bool
    buffer: torch.Tensor
    param_info: List[Dict]
    total_params: int
    chunks: Optional[List[torch.Tensor]] = None

class LayerOffloadConductor:
    def __init__(self, 
                 model: nn.Module, 
                 device: torch.device,
                 max_memory_fraction: float = 0.95,
                 chunk_size_mb: int = 32,
                 enable_logging: bool = False):
        """
        Initialize layer offload conductor with improved memory management
        
        Args:
            model: The model to manage
            device: Target device
            max_memory_fraction: Maximum fraction of GPU memory to use
            chunk_size_mb: Size of memory chunks in MB
            enable_logging: Enable detailed logging
        """
        self.device = device
        self.layers = {}
        self.layer_states = {}
        self.layer_groups = defaultdict(set)
        self.total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        if enable_logging:
            self.logger.setLevel(logging.DEBUG)
        
        # Initialize CUDA streams for overlapped transfers
        self.streams = {
            'h2d': torch.cuda.Stream(priority=-1),  # Host to Device (high priority)
            'd2h': torch.cuda.Stream(priority=0),   # Device to Host (normal priority)
            'compute': torch.cuda.current_stream()  # Main compute stream
        }
        
        # Use dict for O(1) lookups with pre-allocated size
        self.param_cache = dict()
        self.layer_info = {}  # Stores LayerMemoryInfo objects
        
        # Enable performance optimizations
        self._configure_cuda_optimizations()
        
        # Pre-calculate memory thresholds with safety margin
        self.oom_threshold = int(self.total_gpu_memory * max_memory_fraction)
        self.chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
        self.min_pinnable_size = 4 * 1024  # 4KB minimum for pinning
        
        # Initialize memory pool for better allocation
        self._init_memory_pool()
        
        # Register model hooks for activation tracking
        self._register_model_hooks(model)

    def _configure_cuda_optimizations(self):
        """Configure CUDA optimizations based on device capabilities"""
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, running in CPU-only mode")
            return
            
        device_cap = torch.cuda.get_device_capability(self.device)
        
        # Enable TF32 for Ampere+ GPUs
        if device_cap[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmarking for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True
        
        # Set optimal memory allocation strategy
        if hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.memory_stats(self.device)
        
        # Enable tensor cores if available
        if device_cap[0] >= 7:
            torch.set_float32_matmul_precision('high')

    def _init_memory_pool(self):
        """Initialize CUDA memory pool for efficient allocation"""
        if hasattr(torch.cuda, 'memory_pool'):
            self.memory_pool = torch.cuda.memory_pool(self.device)
            # Set memory pool properties
            if hasattr(self.memory_pool, 'set_memory_fraction'):
                self.memory_pool.set_memory_fraction(0.95)  # Reserve some memory for cudnn
        else:
            self.memory_pool = None
            
    def _register_model_hooks(self, model: nn.Module):
        """Register hooks for memory tracking and optimization"""
        def pre_forward_hook(module, input):
            if hasattr(module, '_layer_name'):
                self.before_layer(-1, module._layer_name)
                
        def post_forward_hook(module, input, output):
            if hasattr(module, '_layer_name'):
                self.after_layer(-1, module._layer_name)
        
        # Register hooks on all parameterized layers
        for name, module in model.named_modules():
            if len(list(module.parameters())) > 0:
                module._layer_name = name
                module.register_forward_pre_hook(pre_forward_hook)
                module.register_forward_hook(post_forward_hook)

    def register_layer(self, name: str, module: nn.Module, group: Optional[str] = None):
        """Register layer with optional group"""
        self.layer_states[name] = {
            'module': module,
            'is_active': True
        }
        if group:
            self.layer_groups[group].add(name)
            
    def activate_group(self, group_name: str):
        """Activate all layers in group"""
        if group_name not in self.layer_groups:
            return
            
        for layer_name in self.layer_groups[group_name]:
            if layer_name in self.layer_states:
                self.layer_states[layer_name]['is_active'] = True
                
    def deactivate_group(self, group_name: str):
        """Deactivate all layers in group"""
        if group_name not in self.layer_groups:
            return
            
        for layer_name in self.layer_groups[group_name]:
            if layer_name in self.layer_states:
                self.layer_states[layer_name]['is_active'] = False

    def _analyze_parameters(self, layer: nn.Module) -> Tuple[List[Dict], int, int, List[torch.Tensor]]:
        """Efficiently analyze layer parameters"""
        param_info = []
        total_params = 0
        memory_req = 0
        pinnable_params = []
        
        # Pre-allocate lists for better memory efficiency
        params = list(layer.parameters())
        param_info = [None] * len(params)
        
        for i, p in enumerate(params):
            if not isinstance(p, torch.Tensor) or p.numel() <= 0:
                continue
                
            param_size = p.numel() * p.element_size()
            memory_req += param_size
            total_params += p.numel()
            
            # Fast path for pinnable parameters with vectorized operations
            is_pinnable = (p.is_contiguous() and 
                         not p.is_sparse and 
                         param_size >= self.min_pinnable_size)
                
            if is_pinnable:
                pinnable_params.append(p)
            
            param_info[i] = {
                'numel': p.numel(),
                'shape': p.shape,
                'dtype': p.dtype,
                'id': id(p),
                'requires_grad': p.requires_grad,
                'is_pinnable': is_pinnable,
                'size': param_size,
                'alignment': self._get_tensor_alignment(p)
            }
        
        # Remove None entries from param_info
        param_info = [info for info in param_info if info is not None]
        
        return param_info, total_params, memory_req, pinnable_params
        
    def _get_tensor_alignment(self, tensor: torch.Tensor) -> int:
        """Calculate memory alignment for optimal transfer"""
        if not tensor.is_contiguous():
            return 1
        return tensor.storage_offset() % 512  # Check 512-byte alignment
        
    def _register_small_layer(self, name: str, size: int, param_info: List[Dict], total_params: int):
        """Optimized registration for small layers"""
        buffer = torch.empty(size, dtype=torch.uint8, device='cpu')
        self.layer_info[name] = LayerMemoryInfo(
            size=size,
            pinned=False,
            buffer=buffer,
            param_info=param_info,
            total_params=total_params
        )
        
    def _try_pinned_allocation(self, name: str, pinnable_params: List[torch.Tensor], 
                             param_info: List[Dict], total_params: int) -> bool:
        """Try to allocate pinned memory with error handling"""
        if not pinnable_params:
            return False
            
        try:
            # Calculate total pinnable size
            pinnable_size = sum(p.numel() * p.element_size() for p in pinnable_params)
            
            # Allocate chunks with memory pool if available
            if self.memory_pool is not None:
                chunks = self._allocate_from_pool(pinnable_size)
            else:
                chunks = self._allocate_chunks(pinnable_size)
                
            if not chunks:
                return False
                
            # Create pinned buffer
            pinned_buffer = torch.cat(chunks)[:pinnable_size]
            self.layer_info[name] = LayerMemoryInfo(
                size=pinnable_size,
                pinned=True,
                buffer=pinned_buffer,
                param_info=param_info,
                total_params=total_params,
                chunks=chunks
            )
            return True
            
        except Exception as e:
            self.logger.warning(f"Pinned memory allocation failed for {name}: {e}")
            return False
            
    def _allocate_from_pool(self, total_size: int) -> Optional[List[torch.Tensor]]:
        """Allocate memory from CUDA memory pool"""
        try:
            with torch.cuda.device(self.device):
                chunk = self.memory_pool.allocate(total_size)
                return [chunk]
        except Exception:
            return None
            
    def _register_unpinned(self, name: str, size: int, param_info: List[Dict], total_params: int):
        """Register with unpinned memory, optimized for CPU storage"""
        buffer_size = min(size, self.total_gpu_memory // 16)  # Use smaller buffer
        buffer = torch.empty(buffer_size, dtype=torch.uint8, device='cpu')
        
        self.layer_info[name] = LayerMemoryInfo(
            size=buffer_size,
            pinned=False,
            buffer=buffer,
            param_info=param_info,
            total_params=total_params
        )
        
    def _cache_parameters(self, name: str, layer: nn.Module, total_params: int):
        """Cache layer parameters for faster access"""
        self.param_cache[name] = {
            'param_list': list(layer.parameters()),
            'total_size': total_params,
            'last_access': 0  # For LRU tracking
        }
        
    def _cleanup_failed_registration(self, name: str):
        """Clean up resources after failed layer registration"""
        if name in self.layers:
            del self.layers[name]
        if name in self.layer_states:
            del self.layer_states[name]
        if name in self.layer_info:
            info = self.layer_info[name]
            if info.chunks:
                for chunk in info.chunks:
                    del chunk
            del self.layer_info[name]
        if name in self.param_cache:
            del self.param_cache[name]

    def _allocate_chunks(self, total_size: int) -> List[torch.Tensor]:
        """Helper for chunk allocation with error handling"""
        num_chunks = (total_size + self.chunk_size - 1) // self.chunk_size
        chunks = []
        
        try:
            for _ in range(num_chunks):
                chunk = torch.empty(
                    self.chunk_size,
                    dtype=torch.uint8,
                    pin_memory=True,
                    device='cpu'
                )
                chunks.append(chunk)
            return chunks
        except RuntimeError:
            for chunk in chunks:
                del chunk
            return None

    def move_to_gpu(self, name: str):
        """Optimized GPU transfer with minimal overhead and improved error handling"""
        if name not in self.layers or name not in self.layer_info:
            return
        
        try:
            layer_info = self.layer_info[name]
            cached_params = self.param_cache[name]['param_list']
            
            # Update LRU tracking
            self.param_cache[name]['last_access'] = torch.cuda.current_stream().record_event().elapsed_time(0)
            
            # Check memory availability with safety margin
            required_memory = layer_info.total_params * 2.2  # Account for gradients
            if torch.cuda.memory_allocated() + required_memory > self.oom_threshold:
                self.handle_oom(name)
                if torch.cuda.memory_allocated() + required_memory > self.oom_threshold:
                    raise RuntimeError(f"Insufficient GPU memory for layer {name}")
            
            # Use different transfer strategies based on size
            if layer_info.size < 1024 * 1024:  # Small transfers
                self._transfer_small_layer(name, layer_info, cached_params)
            else:  # Large transfers
                self._transfer_large_layer(name, layer_info, cached_params)
            
            self.layer_states[name] = 'gpu'
            
        except Exception as e:
            self.logger.error(f"Error moving {name} to GPU: {e}")
            self.handle_error(name)
            raise
            
    def _transfer_small_layer(self, name: str, layer_info: LayerMemoryInfo, cached_params: List[torch.Tensor]):
        """Optimized transfer for small layers"""
        with torch.cuda.stream(self.streams['h2d']):
            offset = 0
            for param_info in layer_info.param_info:
                numel = param_info['numel']
                if numel == 0:
                    continue
                    
                param = cached_params[offset // numel]
                if not param.is_cuda:
                    # Fast path for small tensors
                    param.data = layer_info.buffer[offset:offset + numel].view_as(param).to(
                        device=self.device, non_blocking=True
                    )
                offset += numel
                
    def _transfer_large_layer(self, name: str, layer_info: LayerMemoryInfo, cached_params: List[torch.Tensor]):
        """Optimized transfer for large layers with prefetching"""
        with torch.cuda.stream(self.streams['h2d']):
            # Pre-allocate GPU buffers
            gpu_buffers = []
            transfer_sizes = []
            
            # Calculate optimal chunk sizes
            chunk_size = min(layer_info.size // len(cached_params), 64 * 1024 * 1024)  # 64MB chunks
            
            offset = 0
            for param_info in layer_info.param_info:
                numel = param_info['numel']
                if numel == 0:
                    continue
                    
                param = cached_params[offset // numel]
                if not param.is_cuda:
                    # Allocate GPU buffer
                    if param_info['is_pinnable']:
                        # Use pinned memory for faster transfer
                        gpu_buffer = torch.empty_like(param, device=self.device)
                        gpu_buffers.append((gpu_buffer, param))
                        transfer_sizes.append(numel)
                    else:
                        # Direct transfer for non-pinnable parameters
                        param.data = layer_info.buffer[offset:offset + numel].view_as(param).to(
                            device=self.device, non_blocking=True
                        )
                        
                offset += numel
                
            # Perform transfers in parallel
            events = []
            for gpu_buffer, param in gpu_buffers:
                with torch.cuda.stream(self.streams['h2d']):
                    gpu_buffer.copy_(param, non_blocking=True)
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    events.append((event, param, gpu_buffer))
                    
            # Wait for transfers and update parameters
            for event, param, gpu_buffer in events:
                event.synchronize()
                param.data = gpu_buffer
                
    def offload_to_cpu(self, name: str):
        """Optimized CPU offloading with improved memory management"""
        if name not in self.layers or name not in self.layer_info:
            return
        
        try:
            layer_info = self.layer_info[name]
            cached_params = self.param_cache[name]['param_list']
            
            # Use different offload strategies based on size
            if layer_info.size < 1024 * 1024:  # Small layers
                self._offload_small_layer(name, layer_info, cached_params)
            else:  # Large layers
                self._offload_large_layer(name, layer_info, cached_params)
                
            self.layer_states[name] = 'cpu'
            
        except Exception as e:
            self.logger.error(f"Error in offload_to_cpu for {name}: {e}")
            self.handle_error(name)
            
    def _offload_small_layer(self, name: str, layer_info: LayerMemoryInfo, cached_params: List[torch.Tensor]):
        """Optimized offload for small layers"""
        with torch.cuda.stream(self.streams['d2h']):
            offset = 0
            for param_info in layer_info.param_info:
                numel = param_info['numel']
                if numel == 0:
                    continue
                    
                param = cached_params[offset // numel]
                if param.is_cuda:
                    # Direct CPU transfer for small tensors
                    layer_info.buffer[offset:offset + numel].copy_(
                        param.data.to(torch.float16).view(-1),
                        non_blocking=True
                    )
                    param.data = param.data.cpu()
                    
                offset += numel
                
    def _offload_large_layer(self, name: str, layer_info: LayerMemoryInfo, cached_params: List[torch.Tensor]):
        """Optimized offload for large layers with compression"""
        with torch.cuda.stream(self.streams['d2h']):
            offset = 0
            cpu_buffers = []
            
            for param_info in layer_info.param_info:
                numel = param_info['numel']
                if numel == 0:
                    continue
                    
                param = cached_params[offset // numel]
                if param.is_cuda:
                    if param_info['is_pinnable']:
                        # Use pinned memory for faster transfer
                        cpu_buffer = torch.empty(numel, dtype=torch.float16, pin_memory=True)
                        cpu_buffers.append((cpu_buffer, param, offset))
                    else:
                        # Direct transfer with compression
                        layer_info.buffer[offset:offset + numel].copy_(
                            param.data.to(torch.float16).view(-1),
                            non_blocking=True
                        )
                        param.data = param.data.cpu()
                        
                offset += numel
                
            # Perform transfers in parallel
            events = []
            for cpu_buffer, param, buf_offset in cpu_buffers:
                with torch.cuda.stream(self.streams['d2h']):
                    cpu_buffer.copy_(param.data.to(torch.float16).view(-1), non_blocking=True)
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    events.append((event, param, cpu_buffer, buf_offset))
                    
            # Wait for transfers and update buffers
            for event, param, cpu_buffer, buf_offset in events:
                event.synchronize()
                layer_info.buffer[buf_offset:buf_offset + cpu_buffer.numel()].copy_(
                    cpu_buffer, non_blocking=True
                )
                param.data = param.data.cpu()
                
    def handle_oom(self, name: str):
        """Improved OOM handling with smart eviction"""
        torch.cuda.empty_cache()
        
        # Calculate required memory
        required_memory = self.layer_info[name].total_params * 2.2
        
        # Get layers sorted by LRU and size
        gpu_layers = [
            (n, self.layer_info[n].total_params, self.param_cache[n]['last_access'])
            for n, state in self.layer_states.items()
            if state == 'gpu' and n != name
        ]
        
        # Sort by last access time (oldest first) and size (largest first)
        gpu_layers.sort(key=lambda x: (x[2], -x[1]))
        
        # Offload layers until we have enough memory
        freed_memory = 0
        for other_name, size, _ in gpu_layers:
            try:
                self.offload_to_cpu(other_name)
                freed_memory += size * 2.2
                if freed_memory >= required_memory:
                    break
            except Exception as e:
                self.logger.warning(f"Failed to offload layer {other_name}: {e}")
                continue
                
        # Final memory cleanup
        torch.cuda.empty_cache()
        
    def handle_error(self, name: str):
        """Enhanced error recovery"""
        try:
            torch.cuda.empty_cache()
            self.layer_states[name] = 'unknown'
            
            if name in self.layers:
                # Safely move parameters to CPU
                for p in self.layers[name].parameters():
                    if isinstance(p, torch.Tensor) and p.is_cuda:
                        try:
                            p.data = p.data.cpu()
                        except Exception as e:
                            self.logger.error(f"Failed to move parameter to CPU: {e}")
                            
            # Reset CUDA streams
            for stream in self.streams.values():
                try:
                    stream.synchronize()
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error in handle_error for {name}: {e}")
            # Last resort: try to reset CUDA device
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            except:
                pass
    
    def clear(self):
        """Optimized cleanup of all resources"""
        try:
            # Fast synchronization
            current = torch.cuda.current_stream()
            for stream in self.streams.values():
                if stream != current:
                    stream.synchronize()
            
            # Bulk cleanup
            for name in list(self.layers.keys()):
                if self.layer_states.get(name) == 'gpu':
                    try:
                        self.offload_to_cpu(name)
                    except:
                        pass
                
                # Clean up pinned memory chunks
                if name in self.pinned_memory:
                    mem_info = self.pinned_memory[name]
                    if mem_info.get('chunks'):
                        for chunk in mem_info['chunks']:
                            del chunk
            
            # Clear memory
            self.pinned_memory.clear()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error in clear: {e}")
            
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.clear()
        except:
            pass
    
    def load_to_gpu(self, name: str):
        """Optimized GPU loading with minimal synchronization"""
        if name not in self.layers or self.layer_states.get(name) == 'gpu':
            return
            
        try:
            event = torch.cuda.Event(enable_timing=False)
            with torch.cuda.stream(self.streams['h2d']):
                self.move_to_gpu(name)
                event.record()
            
            # Only synchronize if using different stream
            if torch.cuda.current_stream() != self.streams['h2d']:
                torch.cuda.current_stream().wait_event(event)
                
        except Exception as e:
            print(f"Error in load_to_gpu for {name}: {e}")
            self.handle_error(name)

    def before_layer(self, layer_idx: int, name: str):
        """Handle layer activation before computation"""
        if name not in self.layer_states:
            return
            
        try:
            # Load layer to GPU if needed
            if self.layer_states[name] != 'gpu':
                self.load_to_gpu(name)
                
            # Update access time for LRU tracking
            if name in self.param_cache:
                self.param_cache[name]['last_access'] = time.time()
                
            # Pre-allocate buffers if needed
            if name in self.layer_info:
                self.layer_info[name].prepare_buffers()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.handle_oom(name)
                # Retry loading after OOM handling
                self.load_to_gpu(name)
            else:
                raise e

    def after_layer(self, layer_idx: int, name: str):
        """Handle layer cleanup after computation"""
        if name not in self.layer_states:
            return
            
        try:
            # Check memory pressure
            if self.should_offload(name):
                self.offload_to_cpu(name)
                
            # Clear temporary buffers
            if name in self.layer_info:
                self.layer_info[name].clear_temp_buffers()
                
        except Exception as e:
            self.logger.error(f"Error in after_layer for {name}: {e}")
            self.handle_error(name)

    def should_offload(self, name: str) -> bool:
        """Determine if layer should be offloaded based on memory pressure"""
        if not torch.cuda.is_available():
            return False
            
        current_memory = torch.cuda.memory_allocated(self.device)
        if current_memory > self.oom_threshold:
            # Check layer priority and access patterns
            if name in self.param_cache:
                last_access = self.param_cache[name]['last_access']
                size = self.param_cache[name]['total_size']
                
                # Offload if layer is large and not recently accessed
                if size > 100_000_000 and (time.time() - last_access) > 5.0:
                    return True
                    
        return False

def ensure_three_channels(x):
    return x[:3]

def convert_to_bfloat16(x):
    return x.to(torch.bfloat16)

class LayerOffloadManager:
    def __init__(
        self,
        model: nn.Module,
        max_vram_usage: float = 0.8
    ):
        """
        Initialize layer offload manager
        
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

