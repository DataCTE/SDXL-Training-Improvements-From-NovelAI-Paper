import torch
from typing import Dict, Optional, Set
from .Manager import MemoryManager
from .EfficientQuantization import EfficientQuantization
from .EfficientAttention import EfficientAttention
from .layeroffloading import LayerOffloadConductor, LayerOffloadStrategy

class MemoryCoordinator:
    """Coordinates all memory optimization components"""
    def __init__(
        self,
        max_vram_usage: float = 0.8,
        enable_quantization: bool = True,
        enable_efficient_attention: bool = True,
        device: Optional[torch.device] = None,
        strategy: LayerOffloadStrategy = LayerOffloadStrategy.STATIC
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.strategy = strategy
        self.max_vram_usage = max_vram_usage
        
        # Initialize components
        self.memory_manager = MemoryManager(max_vram_usage=max_vram_usage)
        self.quantization = EfficientQuantization(use_layer_scheduling=True) if enable_quantization else None
        self.efficient_attention = EfficientAttention() if enable_efficient_attention else None
        self.layer_conductor = None
        
        # Track active components
        self.active_layer_groups: Set[str] = set()
        self.attention_enabled = enable_efficient_attention
        
    def should_offload_vae(self) -> bool:
        """
        Determine if VAE should be offloaded to CPU based on memory constraints
        and current strategy.
        
        Returns:
            bool: True if VAE should be offloaded to CPU
        """
        # Always offload if using dynamic strategy with low memory threshold
        if self.strategy == LayerOffloadStrategy.DYNAMIC and self.max_vram_usage < 0.6:
            return True
            
        # Check current memory usage
        if hasattr(torch.cuda, 'memory_allocated'):
            current_usage = torch.cuda.memory_allocated(self.device)
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            memory_fraction = current_usage / total_memory
            
            # Offload if memory usage is high
            if memory_fraction > self.max_vram_usage * 0.8:  # 80% of max allowed usage
                return True
                
        return False
        
    def setup_model(self, model: torch.nn.Module):
        """Setup all memory optimizations for model"""
        # Initialize layer conductor with strategy
        self.layer_conductor = LayerOffloadConductor(
            model, 
            self.device,
            max_memory_fraction=0.95 if self.strategy == LayerOffloadStrategy.DYNAMIC else 0.8
        )
        
        # Setup memory management
        self.memory_manager.setup_memory_management(model, self.device)
        
        # Apply quantization if enabled
        if self.quantization:
            self.quantization.quantize_model(model, layer_group="main")
            
        # Replace attention layers if enabled
        if self.efficient_attention:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.MultiheadAttention):
                    self.efficient_attention.replace_attention(module)
                    
    def activate_layer_group(self, group_name: str):
        """Coordinate activation of layer group across components"""
        if group_name in self.active_layer_groups:
            return
            
        # Activate in quantization
        if self.quantization:
            self.quantization.activate_layer_group(group_name)
            
        # Update layer conductor
        if self.layer_conductor:
            self.layer_conductor.activate_group(group_name)
            
        self.active_layer_groups.add(group_name)
        
    def deactivate_layer_group(self, group_name: str):
        """Coordinate deactivation of layer group"""
        if group_name not in self.active_layer_groups:
            return
            
        if self.quantization:
            self.quantization.deactivate_layer_group(group_name)
            
        if self.layer_conductor:
            self.layer_conductor.deactivate_group(group_name)
            
        self.active_layer_groups.remove(group_name)
        
    def get_memory_stats(self) -> Dict:
        """Get combined memory statistics"""
        stats = {
            'manager': self.memory_manager.get_current_usage(),
            'peak_memory': torch.cuda.max_memory_allocated(self.device) / (1024**3)
        }
        
        if self.quantization:
            stats['quantization'] = self.quantization.get_memory_savings()
            
        return stats 