import torch
import torch.cuda
import torch.cuda.memory
import gc
from typing import Dict, List, Optional
import torch.nn as nn
from memory.quantization import is_quantized_parameter, get_offload_tensor_bytes
from memory.EfficientQuantization import MemoryEfficientQuantization
from memory.layeroffloading import LayerOffloadConductor, LayerOffloadStrategy

class MemoryManager:
    def __init__(self, max_vram_usage: float = 0.8):
        self.peak_memory = 0
        self.current_memory = 0
        self.oom_count = 0
        self.registered_modules: Dict[str, nn.Module] = {}
        self.module_memory_usage: Dict[str, int] = {}
        
        # Memory management components
        self.quantization = MemoryEfficientQuantization()
        self.max_vram_usage = max_vram_usage
        self.layer_conductor: Optional[LayerOffloadConductor] = None
        self.layer_strategy: Optional[LayerOffloadStrategy] = None
        
    def setup_memory_management(self, model: nn.Module, device: torch.device):
        """Initialize memory management components"""
        # Setup layer offloading
        self.layer_conductor = LayerOffloadConductor(model, device)
        self.layer_strategy = LayerOffloadStrategy(model, self.max_vram_usage)
        
        # Setup quantization
        self.quantization.setup_quantization(model, device, torch.bfloat16)
        
    def register_module(self, name: str, module: nn.Module):
        """Register a module for memory tracking"""
        self.registered_modules[name] = module
        
        # Calculate memory usage including quantized parameters
        if is_quantized_parameter(module, "weight"):
            self.module_memory_usage[name] = get_offload_tensor_bytes(module)
        else:
            self.module_memory_usage[name] = sum(
                p.numel() * p.element_size() 
                for p in module.parameters()
            )
            
        # Register with layer conductor if available
        if self.layer_conductor is not None:
            self.layer_conductor.register_layer(name, module)
            
    def before_layer_computation(self, layer_name: str, phase: str = 'forward'):
        """Prepare memory for layer computation"""
        if self.layer_strategy and self.layer_conductor:
            # Get required layers
            required_layers = self.layer_strategy.get_required_layers(layer_name, phase)
            
            # Get layers to offload
            to_offload = self.layer_strategy.suggest_offload(required_layers)
            
            # Perform offloading
            for name in to_offload:
                self.layer_conductor.offload_to_cpu(name)
                self.layer_strategy.update_vram_usage(name, False)
            
            # Load required layers
            for name in required_layers:
                self.layer_conductor.load_to_gpu(name)
                self.layer_strategy.update_vram_usage(name, True)
                
            # Synchronize transfers
            self.layer_conductor.synchronize()
            
    def after_layer_computation(self, layer_name: str):
        """Cleanup after layer computation"""
        if self.layer_conductor:
            self.layer_conductor.after_layer(None, layer_name)
        self.clear_cache()
        
    def update(self):
        """Update memory tracking"""
        self.current_memory = torch.cuda.memory_allocated()
        self.peak_memory = max(self.peak_memory, self.current_memory)
        
    def clear_cache(self):
        """Clear CUDA cache and collect garbage"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
    def handle_oom(self):
        """Handle out of memory error"""
        self.oom_count += 1
        self.clear_cache()
        
        # Try emergency memory recovery
        if self.layer_conductor and self.oom_count > 2:
            # Offload all non-essential layers
            current_layers = set(self.layer_conductor.layer_states.keys())
            self.layer_conductor.synchronize()
            for name in current_layers:
                self.layer_conductor.offload_to_cpu(name)
                
        return self.oom_count > 3  # Return True if OOM is persistent
        
    def get_module_memory(self, name: str) -> int:
        """Get memory usage for a specific module"""
        return self.module_memory_usage.get(name, 0)
        
    def get_current_usage(self) -> float:
        """Get current total memory usage in bytes"""
        self.update()
        return self.current_memory
        
    def get_peak_usage(self) -> float:
        """Get peak memory usage in bytes"""
        return self.peak_memory
        
    def log_memory_stats(self):
        """Log detailed memory statistics"""
        print(f"Memory Stats:")
        print(f"  Current: {self.current_memory/1e9:.1f}GB")
        print(f"  Peak: {self.peak_memory/1e9:.1f}GB")
        print(f"  OOM Count: {self.oom_count}")
        print(f"  VRAM Usage: {(self.current_memory/torch.cuda.get_device_properties(0).total_memory)*100:.1f}%")
        
        print("\nModule Memory Usage:")
        for name, usage in self.module_memory_usage.items():
            status = "GPU" if (self.layer_conductor and 
                             self.layer_conductor.layer_states.get(name, {}).get('location') == 'gpu') else "CPU"
            print(f"  {name}: {usage/1e6:.1f}MB ({status})")
