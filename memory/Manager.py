import torch
import torch.cuda
import torch.cuda.memory
import gc
from typing import Dict, List, Optional, Tuple, Set
import torch.nn as nn
from dataclasses import dataclass
import logging
import psutil
import time
from memory.quantization import is_quantized_parameter, get_offload_tensor_bytes
from memory.EfficientQuantization import EfficientQuantization
from memory.layeroffloading import LayerOffloadConductor, LayerOffloadManager
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Statistics for memory usage"""
    gpu_allocated: int = 0
    gpu_cached: int = 0
    gpu_peak: int = 0
    cpu_used: int = 0
    cpu_peak: int = 0
    timestamp: float = 0.0

class MemoryManager:
    """Enhanced memory manager with improved monitoring and management"""
    
    def __init__(
        self, 
        max_vram_usage: float = 0.8,
        enable_monitoring: bool = True,
        monitoring_interval: float = 1.0,
        enable_logging: bool = True
    ):
        """
        Initialize memory manager
        
        Args:
            max_vram_usage: Maximum fraction of VRAM to use
            enable_monitoring: Whether to enable memory monitoring
            monitoring_interval: Interval between memory checks in seconds
            enable_logging: Whether to enable detailed logging
        """
        # Memory tracking
        self.stats_history: List[MemoryStats] = []
        self.peak_memory = 0
        self.current_memory = 0
        self.oom_count = 0
        self.last_check = time.time()
        
        # Configuration
        self.max_vram_usage = max_vram_usage
        self.monitoring_interval = monitoring_interval
        self.enable_monitoring = enable_monitoring
        self.enable_logging = enable_logging
        
        # Module tracking
        self.registered_modules: Dict[str, nn.Module] = {}
        self.module_memory_usage: Dict[str, int] = {}
        self.active_modules: Set[str] = set()
        
        # Memory management components
        self.quantization = EfficientQuantization()
        self.layer_conductor: Optional[LayerOffloadConductor] = None
        self.layer_strategy: Optional[LayerOffloadManager] = None
        
        # Add efficient memory pools
        self.memory_pools = {}
        self.current_workspace = None
        
        # Pre-allocate memory pools for common sizes
        self.initialize_memory_pools()
        
        if enable_logging:
            self._setup_logging()
            
        # Add group tracking
        self.layer_groups = defaultdict(set)
        self.active_groups = set()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def setup_memory_management(
        self, 
        model: nn.Module, 
        device: torch.device,
        train_dtype: Optional[torch.dtype] = None
    ) -> None:
        """
        Initialize memory management components with error handling
        
        Args:
            model: The model to manage
            device: Target device
            train_dtype: Training data type (defaults to model dtype)
        """
        try:
            # Setup layer offloading with monitoring
            self.layer_conductor = LayerOffloadConductor(
                model, 
                device,
                max_memory_fraction=self.max_vram_usage,
                enable_logging=self.enable_logging
            )
            self.layer_strategy = LayerOffloadManager(model, self.max_vram_usage)
            
            # Setup quantization if train_dtype is provided
            if train_dtype is not None:
                self.quantization.setup_quantization(
                    model, 
                    device, 
                    train_dtype,
                    enable_logging=self.enable_logging
                )
                
            # Initial memory check
            self.update_memory_stats()
            
        except Exception as e:
            logger.error(f"Error setting up memory management: {e}")
            raise
            
    def register_module(
        self, 
        name: str, 
        module: nn.Module,
        track_activations: bool = False,
        group: Optional[str] = None
    ) -> None:
        """
        Register a module for memory tracking with activation monitoring
        
        Args:
            name: Module name
            module: The module to register
            track_activations: Whether to track activation memory
            group: Optional group name
        """
        try:
            self.registered_modules[name] = module
            
            # Calculate memory usage including quantized parameters
            memory_usage = 0
            if is_quantized_parameter(module, "weight"):
                memory_usage = get_offload_tensor_bytes(module)
            else:
                memory_usage = sum(
                    p.numel() * p.element_size() 
                    for p in module.parameters()
                )
                
            # Add estimated activation memory if tracking
            if track_activations:
                memory_usage += self._estimate_activation_memory(module)
                
            self.module_memory_usage[name] = memory_usage
            
            # Register with layer conductor
            if self.layer_conductor is not None:
                self.layer_conductor.register_layer(name, module)
                
            if self.enable_logging:
                logger.info(f"Registered module {name} with {memory_usage/1e6:.1f}MB memory usage")
                
            if group:
                self.layer_groups[group].add(name)
                
        except Exception as e:
            logger.error(f"Error registering module {name}: {e}")
            raise
            
    def _estimate_activation_memory(self, module: nn.Module) -> int:
        """Estimate activation memory for a module"""
        try:
            total_params = sum(p.numel() for p in module.parameters())
            # Rough estimation based on parameter count and typical activation sizes
            return total_params * 4  # Assume float32 activations
        except Exception:
            return 0
            
    def before_layer_computation(
        self, 
        layer_name: str, 
        phase: str = 'forward'
    ) -> None:
        """
        Prepare memory for layer computation with prefetching
        
        Args:
            layer_name: Name of the layer
            phase: Computation phase ('forward' or 'backward')
        """
        if not self._is_layer_active(layer_name):
            return
            
        try:
            if self.layer_strategy and self.layer_conductor:
                # Update memory stats before computation
                self.update_memory_stats()
                
                # Get required layers with prefetching
                required_layers = self.layer_strategy.get_required_layers(layer_name, phase)
                prefetch_layers = self._get_prefetch_layers(required_layers)
                
                # Get layers to offload
                to_offload = self.layer_strategy.suggest_offload(
                    required_layers + prefetch_layers
                )
                
                # Perform offloading and loading in parallel streams
                self._parallel_memory_transfer(to_offload, required_layers, prefetch_layers)
                
                # Track active modules
                self.active_modules.add(layer_name)
                
        except Exception as e:
            logger.error(f"Error in before_layer_computation for {layer_name}: {e}")
            self.handle_error(layer_name)
            
    def _get_prefetch_layers(self, required_layers: List[str]) -> List[str]:
        """Determine which layers to prefetch based on computation pattern"""
        try:
            # Simple prefetching: get next few layers in sequence
            all_layers = list(self.registered_modules.keys())
            if not required_layers:
                return []
                
            current_idx = all_layers.index(required_layers[-1])
            next_layers = all_layers[current_idx + 1:current_idx + 3]
            return [l for l in next_layers if l not in required_layers]
            
        except Exception:
            return []
            
    def _parallel_memory_transfer(
        self,
        to_offload: List[str],
        required_layers: List[str],
        prefetch_layers: List[str]
    ) -> None:
        """Perform memory transfers in parallel streams"""
        try:
            if not self.layer_conductor:
                return
                
            # Offload in device-to-host stream
            for name in to_offload:
                self.layer_conductor.offload_to_cpu(name)
                self.layer_strategy.update_vram_usage(name, False)
                self.active_modules.discard(name)
                
            # Load required layers in host-to-device stream
            for name in required_layers:
                self.layer_conductor.load_to_gpu(name)
                self.layer_strategy.update_vram_usage(name, True)
                
            # Prefetch in separate stream if memory available
            if self.current_memory < self.max_vram_usage * torch.cuda.get_device_properties(0).total_memory:
                for name in prefetch_layers:
                    self.layer_conductor.prefetch_to_gpu(name)
                    
        except Exception as e:
            logger.error(f"Error in parallel memory transfer: {e}")
            
    def after_layer_computation(self, layer_name: str) -> None:
        """Cleanup after layer computation with memory tracking"""
        try:
            if self.layer_conductor:
                self.layer_conductor.after_layer(None, layer_name)
                
            self.active_modules.discard(layer_name)
            self.update_memory_stats()
            self.clear_cache()
            
        except Exception as e:
            logger.error(f"Error in after_layer_computation for {layer_name}: {e}")
            self.handle_error(layer_name)
            
    def update_memory_stats(self) -> None:
        """Update memory statistics with monitoring"""
        current_time = time.time()
        
        # Only update at monitoring interval
        if not self.enable_monitoring or (current_time - self.last_check) < self.monitoring_interval:
            return
            
        try:
            # Get process memory info in a platform-independent way
            process = psutil.Process()
            memory_info = process.memory_info()
            
            stats = MemoryStats(
                gpu_allocated=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                gpu_cached=torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                gpu_peak=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                cpu_used=memory_info.rss,  # Resident Set Size - cross-platform
                cpu_peak=getattr(memory_info, 'peak_wset', memory_info.rss),  # Fallback to RSS if peak not available
                timestamp=current_time
            )
            
            self.stats_history.append(stats)
            self.current_memory = stats.gpu_allocated
            self.peak_memory = max(self.peak_memory, stats.gpu_peak)
            self.last_check = current_time
            
            # Log if memory usage is high
            if (torch.cuda.is_available() and 
                self.enable_logging and 
                self.current_memory > 0.9 * self.max_vram_usage * torch.cuda.get_device_properties(0).total_memory):
                logger.warning("High memory usage detected")
                self.log_memory_stats()
                
        except Exception as e:
            logger.error(f"Error updating memory stats: {e}")
            
    def clear_cache(self) -> None:
        """Clear CUDA cache and collect garbage with monitoring"""
        try:
            # Record memory before cleanup
            pre_clean_memory = torch.cuda.memory_allocated()
            
            # Perform cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
            # Record memory after cleanup
            post_clean_memory = torch.cuda.memory_allocated()
            
            if self.enable_logging:
                freed_memory = pre_clean_memory - post_clean_memory
                if freed_memory > 0:
                    logger.info(f"Cleared {freed_memory/1e6:.1f}MB of CUDA memory")
                    
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            
    def handle_oom(self) -> bool:
        """
        Handle out of memory error with recovery strategies
        
        Returns:
            bool: True if OOM is persistent and unrecoverable
        """
        self.oom_count += 1
        logger.warning(f"OOM detected (count: {self.oom_count})")
        
        try:
            # Clear cache first
            self.clear_cache()
            
            # Try emergency memory recovery
            if self.layer_conductor and self.oom_count > 2:
                # Offload all non-essential layers
                current_layers = set(self.layer_conductor.layer_states.keys())
                essential_layers = self.active_modules
                
                self.layer_conductor.synchronize()
                for name in current_layers - essential_layers:
                    self.layer_conductor.offload_to_cpu(name)
                    
                # Try to compact memory if available
                if hasattr(torch.cuda, 'memory_stats'):
                    torch.cuda.memory_stats()
                    if hasattr(torch.cuda, 'memory_summary'):
                        logger.info(torch.cuda.memory_summary())
                        
            # Check if OOM is persistent
            is_persistent = self.oom_count > 3
            if is_persistent:
                logger.error("Persistent OOM detected - memory state may be unrecoverable")
                self.log_memory_stats()
                
            return is_persistent
            
        except Exception as e:
            logger.error(f"Error in OOM handling: {e}")
            return True
            
    def handle_error(self, name: str) -> None:
        """Enhanced error handling with recovery attempts"""
        try:
            logger.error(f"Error detected for module {name}")
            
            # Clear CUDA cache and synchronize
            self.clear_cache()
            
            # Try to recover module state
            if name in self.registered_modules:
                module = self.registered_modules[name]
                
                # Move parameters to CPU
                for param in module.parameters():
                    if isinstance(param, torch.Tensor) and param.is_cuda:
                        try:
                            param.data = param.data.cpu()
                        except Exception as e:
                            logger.error(f"Failed to move parameter to CPU: {e}")
                            
            # Update memory tracking
            self.update_memory_stats()
            
        except Exception as e:
            logger.error(f"Error in error handling for {name}: {e}")
            
    def get_module_memory(self, name: str) -> int:
        """Get current memory usage for a specific module"""
        return self.module_memory_usage.get(name, 0)
        
    def get_current_usage(self) -> float:
        """Get current total memory usage in bytes"""
        self.update_memory_stats()
        return self.current_memory
        
    def get_peak_usage(self) -> float:
        """Get peak memory usage in bytes"""
        return self.peak_memory
        
    def log_memory_stats(self) -> None:
        """Log detailed memory statistics"""
        try:
            stats = self.stats_history[-1] if self.stats_history else None
            if not stats:
                return
                
            logger.info("Memory Statistics:")
            logger.info(f"  GPU Allocated: {stats.gpu_allocated/1e9:.1f}GB")
            logger.info(f"  GPU Cached: {stats.gpu_cached/1e9:.1f}GB")
            logger.info(f"  GPU Peak: {stats.gpu_peak/1e9:.1f}GB")
            logger.info(f"  CPU Used: {stats.cpu_used/1e9:.1f}GB")
            logger.info(f"  CPU Peak: {stats.cpu_peak/1e9:.1f}GB")
            logger.info(f"  OOM Count: {self.oom_count}")
            
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            usage_percent = (self.current_memory/total_gpu_memory) * 100
            logger.info(f"  VRAM Usage: {usage_percent:.1f}%")
            
            logger.info("\nModule Memory Usage:")
            for name, usage in self.module_memory_usage.items():
                status = "GPU" if name in self.active_modules else "CPU"
                logger.info(f"  {name}: {usage/1e6:.1f}MB ({status})")
                
        except Exception as e:
            logger.error(f"Error logging memory stats: {e}")
            
    def get_memory_timeline(self) -> List[Tuple[float, float]]:
        """Get memory usage timeline for visualization"""
        return [(stats.timestamp, stats.gpu_allocated/1e9) for stats in self.stats_history]
        
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.clear_cache()
        except:
            pass
        
    def initialize_memory_pools(self):
        """Pre-allocate memory pools for common tensor sizes"""
        common_sizes = [
            (1024, 1024),  # 1MB
            (2048, 2048),  # 4MB
            (4096, 4096)   # 16MB
        ]
        for size in common_sizes:
            self.memory_pools[size] = torch.cuda.Stream()
            with torch.cuda.stream(self.memory_pools[size]):
                # Pre-allocate workspace
                torch.empty(size, device='cuda')
        
    def _is_layer_active(self, layer_name: str) -> bool:
        """Check if layer is in active group"""
        for group in self.active_groups:
            if layer_name in self.layer_groups[group]:
                return True
        return False
