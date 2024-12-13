import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from PIL import Image
from pathlib import Path
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Callable, Union
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from torchvision import transforms
from accelerate import Accelerator
from adamw_bf16 import AdamWBF16
import glob
import os
from tqdm import tqdm
import wandb
from torch.nn.utils import clip_grad_norm_
import shutil
import argparse
from safetensors.torch import load_file
import signal
import sys
import torch.cuda.streams as streams
import json
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from math import ceil
from torch.profiler import profile, record_function, ProfilerActivity
import contextlib
import time
import re
from collections import defaultdict

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear8bitLt, LinearNf4  # Add LinearNf4
    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False
    Linear8bitLt = None
    LinearNf4 = None  # Add this
    bnb = None

# Add torch_gc import for memory management
try:
    from torch.cuda import empty_cache as torch_gc
except ImportError:
    def torch_gc():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()



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


@dataclass
class ImageBucket:
    width: int
    height: int
    items: List = None
    
    def __post_init__(self):
        self.aspect_ratio = self.width / self.height
        if self.items is None:
            self.items = []

class AspectRatioBucket:
    def __init__(
        self,
        max_image_size: Tuple[int, int] = (768, 1024),
        max_dim: int = 1024,
        bucket_step: int = 64
    ):
        self.max_width, self.max_height = max_image_size
        self.max_dim = max_dim
        self.bucket_step = bucket_step
        self.buckets: List[ImageBucket] = []
        self._generate_buckets()
        
    def _generate_buckets(self):
        """Generate bucket resolutions following section 4.1.2"""
        # Generate width-first buckets
        width = 256
        while width <= self.max_dim:
            # Find largest height that satisfies constraints
            height = min(
                self.max_dim,
                math.floor(self.max_width * self.max_height / width)
            )
            self.buckets.append(ImageBucket(
                width=width,
                height=height
            ))
            width += self.bucket_step
            
        # Generate height-first buckets
        height = 256
        while height <= self.max_dim:
            width = min(
                self.max_dim,
                math.floor(self.max_width * self.max_height / height)
            )
            # Skip if bucket already exists
            if not any(b.width == width and b.height == height for b in self.buckets):
                self.buckets.append(ImageBucket(
                    width=width,
                    height=height
                ))
            height += self.bucket_step
            
        # Add standard square bucket
        if not any(b.width == 1024 and b.height == 1024 for b in self.buckets):
            self.buckets.append(ImageBucket(
                width=1024,
                height=1024
            ))

    def find_bucket(self, width: int, height: int) -> Optional[ImageBucket]:
        """Find best fitting bucket for given image dimensions"""
        image_aspect = width / height
        log_aspects = np.log([b.aspect_ratio for b in self.buckets])
        log_image_aspect = np.log(image_aspect)
        
        # Find closest bucket in log-space
        idx = np.argmin(np.abs(log_aspects - log_image_aspect))
        return self.buckets[idx]

class TextEmbedder:
    def __init__(
        self,
        device: torch.device,
        tokenizer_paths: Dict[str, str] = {
            "base": "openai/clip-vit-large-patch14",
            "large": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        }
    ):
        self.device = device
        self.max_length = 77
        
        # Load tokenizers (remove subfolder paths)
        self.tokenizers = {
            "base": CLIPTokenizer.from_pretrained(tokenizer_paths["base"]),
            "large": CLIPTokenizer.from_pretrained(tokenizer_paths["large"])
        }
        
        # Load text encoders with bfloat16 (remove subfolder paths)
        self.text_encoders = {
            "base": CLIPTextModel.from_pretrained(tokenizer_paths["base"]).to(device).to(torch.bfloat16),
            "large": CLIPTextModel.from_pretrained(tokenizer_paths["large"]).to(device).to(torch.bfloat16)
        }
        
        for encoder in self.text_encoders.values():
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def __call__(self, prompt: str) -> Dict[str, torch.Tensor]:
        # Tokenize
        tokens = {
            k: tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            for k, tokenizer in self.tokenizers.items()
        }
        
        # Generate embeddings
        embeds = {}
        for k, encoder in self.text_encoders.items():
            # Move tokens to GPU but keep as int64 for embedding layer
            tokens_gpu = {key: val.to(self.device) for key, val in tokens[k].items()}
            
            # Generate embeddings
            output = encoder(**tokens_gpu)
            
            # Ensure text embeddings have shape [batch_size, seq_len, hidden_dim]
            last_hidden_state = output.last_hidden_state
            if last_hidden_state.dim() == 2:
                last_hidden_state = last_hidden_state.unsqueeze(0)
            
            # Ensure pooled embeddings have shape [batch_size, hidden_dim]
            pooled_output = output.pooler_output
            if pooled_output.dim() == 1:
                pooled_output = pooled_output.unsqueeze(0)
            elif pooled_output.dim() == 3:
                pooled_output = pooled_output.squeeze(1)
            
            # Keep embeddings on CPU with consistent dimensions
            embeds[f"{k}_text_embeds"] = last_hidden_state.cpu()  # [batch_size, seq_len, hidden_dim]
            embeds[f"{k}_pooled_embeds"] = pooled_output.cpu()   # [batch_size, hidden_dim]
            
        return embeds

class TagCategory:
    """Tag categories with importance weights"""
    SUBJECT = "subject"
    STYLE = "style"
    QUALITY = "quality"
    COMPOSITION = "composition"
    COLOR = "color"
    LIGHTING = "lighting"
    TECHNICAL = "technical"
    MEDIUM = "medium"
    ARTIST = "artist"

class OptimizedTagClassifier:
    """Memory and compute optimized tag classifier"""
    def __init__(self):
        # Pre-compile all patterns into a single regex per category for faster matching
        self.category_patterns = {
            TagCategory.SUBJECT: (
                r"(person|man|woman|girl|boy|child|people|"
                r"landscape|nature|city|building|"
                r"animal|cat|dog|bird|"
                r"object|item|thing)",
                1.2
            ),
            TagCategory.STYLE: (
                r"(realistic|photorealistic|abstract|"
                r"anime|cartoon|digital art|"
                r"painting|sketch|drawing)",
                1.1
            ),
            TagCategory.QUALITY: (
                r"(high quality|masterpiece|best quality|"
                r"detailed|intricate|sharp|"
                r"professional|award winning)",
                0.9
            ),
            TagCategory.COMPOSITION: (
                r"(portrait|close-up|wide shot|"
                r"symmetrical|balanced|centered|"
                r"dynamic|action|motion)",
                1.0
            ),
            TagCategory.COLOR: (
                r"(colorful|vibrant|monochrome|"
                r"red|blue|green|yellow|purple|pink|"
                r"dark|light|bright|muted)",
                0.8
            ),
            TagCategory.LIGHTING: (
                r"(sunlight|natural light|artificial light|"
                r"dramatic lighting|soft lighting|"
                r"shadow|highlight|contrast)",
                1.0
            ),
            TagCategory.TECHNICAL: (
                r"(8k|4k|uhd|hdr|"
                r"raw photo|dslr|bokeh|"
                r"lens|camera|settings)",
                0.7
            ),
            TagCategory.MEDIUM: (
                r"(photograph|digital|traditional|"
                r"oil|watercolor|acrylic|"
                r"pencil|charcoal|ink)",
                0.9
            ),
            TagCategory.ARTIST: (
                r"(by \w+|style of \w+|inspired by|"
                r"artist:|photographer:|"
                r"school|movement|period)",
                0.8
            )
        }
        
        # Compile patterns once
        self.compiled_patterns = {
            category: (re.compile(pattern, re.IGNORECASE), importance)
            for category, (pattern, importance) in self.category_patterns.items()
        }
        
        # Cache for tag classifications
        self.classification_cache = {}
    
    def classify_tag(self, tag: str) -> Tuple[str, float]:
        """Classify a tag with caching for performance"""
        # Check cache first
        if tag in self.classification_cache:
            return self.classification_cache[tag]
        
        # Match against compiled patterns
        for category, (pattern, importance) in self.compiled_patterns.items():
            if pattern.search(tag):
                result = (category, importance)
                self.classification_cache[tag] = result
                return result
        
        # Cache and return default for unknown tags
        result = ("other", 1.0)
        self.classification_cache[tag] = result
        return result

class FastTagWeighter:
    """Performance optimized tag weighting system"""
    def __init__(
        self,
        min_weight: float = 0.1,
        max_weight: float = 2.0,
        default_weight: float = 1.0,
        smoothing_factor: float = 0.1,
        rarity_scale: float = 0.5,
        cache_size: int = 10000
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.default_weight = default_weight
        self.smoothing_factor = smoothing_factor
        self.rarity_scale = rarity_scale
        
        # Use numpy arrays for faster computation
        self.tag_counts = np.zeros(cache_size, dtype=np.int32)
        self.category_counts = np.zeros(len(TagCategory.__dict__) - 2, dtype=np.int32)
        self.total_samples = 0
        
        # Fast lookup dictionaries
        self.tag_to_idx = {}
        self.next_tag_idx = 0
        self.category_to_idx = {
            category: idx for idx, category in enumerate(
                [attr for attr in dir(TagCategory) if not attr.startswith('_')]
            )
        }
        
        # Initialize classifier
        self.classifier = OptimizedTagClassifier()
        
        # Use numpy arrays for moving averages
        self.tag_moving_avg = np.ones(cache_size, dtype=np.float32) * default_weight
        self.category_moving_avg = np.ones(len(self.category_to_idx), dtype=np.float32) * default_weight
        
        # Weight computation cache
        self.weight_cache = {}
        self.weight_cache_size = 1000
        self.weight_cache_hits = 0
        self.weight_cache_misses = 0
    
    def _get_tag_idx(self, tag: str) -> int:
        """Get or create index for tag"""
        if tag not in self.tag_to_idx:
            if self.next_tag_idx >= len(self.tag_counts):
                # Double array size if needed
                self.tag_counts = np.pad(self.tag_counts, (0, len(self.tag_counts)))
                self.tag_moving_avg = np.pad(
                    self.tag_moving_avg, 
                    (0, len(self.tag_moving_avg)),
                    constant_values=self.default_weight
                )
            self.tag_to_idx[tag] = self.next_tag_idx
            self.next_tag_idx += 1
        return self.tag_to_idx[tag]
    
    @torch.jit.script
    def _compute_weights_fast(
        self,
        tag_freqs: torch.Tensor,
        category_freqs: torch.Tensor,
        base_importance: torch.Tensor,
        tag_counts: torch.Tensor,
        category_counts: torch.Tensor,
        rarity_scale: float,
        smoothing_factor: float
    ) -> torch.Tensor:
        """JIT-compiled weight computation"""
        # Calculate frequency weights
        tag_weights = 1.0 / (tag_freqs + smoothing_factor)
        category_weights = 1.0 / (category_freqs + smoothing_factor)
        
        # Calculate rarity bonus
        rarity_bonus = 1.0 + rarity_scale * (1.0 - tag_counts / category_counts.unsqueeze(1))
        
        # Combine weights
        combined_weights = (
            base_importance * 
            (tag_weights * 0.6 + category_weights * 0.4) * 
            rarity_bonus
        )
        
        return combined_weights
    
    def update_frequencies(self, tags: List[str]):
        """Vectorized frequency update"""
        self.total_samples += 1
        
        # Convert tags to indices
        tag_indices = np.array([self._get_tag_idx(tag) for tag in tags])
        
        # Get categories and importance values
        categories = []
        importance_values = []
        for tag in tags:
            category, importance = self.classifier.classify_tag(tag)
            categories.append(self.category_to_idx[category])
            importance_values.append(importance)
        
        category_indices = np.array(categories)
        
        # Vectorized updates
        np.add.at(self.tag_counts, tag_indices, 1)
        np.add.at(self.category_counts, category_indices, 1)
        
        # Update moving averages in batches
        tag_freqs = self.tag_counts[tag_indices] / self.total_samples
        category_freqs = self.category_counts[category_indices] / self.total_samples
        
        current_tag_weights = 1.0 / (tag_freqs + self.smoothing_factor)
        current_category_weights = 1.0 / (category_freqs + self.smoothing_factor)
        
        # Vectorized moving average update
        self.tag_moving_avg[tag_indices] = (
            self.tag_moving_avg[tag_indices] * (1 - self.smoothing_factor) +
            current_tag_weights * self.smoothing_factor
        )
        
        np.add.at(
            self.category_moving_avg,
            category_indices,
            (current_category_weights - self.category_moving_avg[category_indices]) * self.smoothing_factor
        )
        
        # Clear weight cache when frequencies update
        if len(self.weight_cache) > self.weight_cache_size:
            self.weight_cache.clear()
    
    def get_tag_weight(self, tag: str) -> float:
        """Get cached or compute tag weight"""
        # Check cache
        if tag in self.weight_cache:
            self.weight_cache_hits += 1
            return self.weight_cache[tag]
        
        self.weight_cache_misses += 1
        
        # Get tag index and category
        tag_idx = self._get_tag_idx(tag)
        category, importance = self.classifier.classify_tag(tag)
        category_idx = self.category_to_idx[category]
        
        # Convert to tensors for fast computation
        tag_freq = torch.tensor(self.tag_counts[tag_idx] / self.total_samples)
        category_freq = torch.tensor(self.category_counts[category_idx] / self.total_samples)
        
        # Compute weight using JIT-compiled function
        weight = self._compute_weights_fast(
            tag_freq.unsqueeze(0),
            category_freq.unsqueeze(0),
            torch.tensor([importance]),
            torch.tensor([self.tag_counts[tag_idx]]),
            torch.tensor([self.category_counts[category_idx]]),
            self.rarity_scale,
            self.smoothing_factor
        )[0].item()
        
        # Cache result
        self.weight_cache[tag] = weight
        
        # Clear cache if too large
        if len(self.weight_cache) > self.weight_cache_size:
            self.weight_cache.clear()
        
        return weight
    
    @torch.jit.script
    def _compute_combined_weight(self, weights: torch.Tensor) -> float:
        """JIT-compiled weight combination"""
        log_weights = torch.log(weights)
        return torch.exp(log_weights.mean()).item()
    
    def get_weights(self, tags: List[str]) -> float:
        """Get combined weights using vectorized operations"""
        if not tags:
            return self.default_weight
        
        # Convert to tensor for fast computation
        weights = torch.tensor([self.get_tag_weight(tag) for tag in tags])
        return self._compute_combined_weight(weights)
    
    def print_stats(self):
        """Print cache performance statistics"""
        total = self.weight_cache_hits + self.weight_cache_misses
        hit_rate = self.weight_cache_hits / total if total > 0 else 0
        print(f"Weight cache hit rate: {hit_rate:.2%}")
        print(f"Unique tags: {self.next_tag_idx}")
        print(f"Cache size: {len(self.weight_cache)}")

def parse_tags(caption: str) -> List[str]:
    """Enhanced tag parsing with better handling of special cases"""
    # Convert to lowercase and split on commas
    parts = caption.lower().split(',')
    
    # Clean and normalize tags
    tags = []
    for part in parts:
        # Basic cleaning
        tag = part.strip()
        
        # Skip empty tags
        if not tag:
            continue
            
        # Handle special cases
        if ':' in tag:  # Handle key:value pairs
            key, value = tag.split(':', 1)
            tags.extend([key.strip(), value.strip()])
        elif 'style of' in tag:  # Handle artist styles
            artist = tag.replace('style of', '').strip()
            tags.extend(['style', artist])
        elif tag.startswith(('by ', 'artist ')):  # Handle artist attribution
            artist = tag.replace('by', '').replace('artist', '').strip()
            tags.extend(['artist', artist])
        else:
            tags.append(tag)
    
    return tags

class MultiDirectoryImageDataset(Dataset):
    """Dataset that handles multiple image directories with efficient loading"""
    
    def __init__(
        self,
        image_dirs: List[str],
        transform: Optional[Callable] = None,
        cache_dir: str = "latent_cache",
        text_cache_dir: str = "text_cache",
        min_size: int = 512,
        max_size: int = 2048,
        aspect_ratio_range: Tuple[float, float] = (0.5, 2.0)
    ):
        self.transform = transform
        self.cache_dir = Path(cache_dir)
        self.text_cache_dir = Path(text_cache_dir)
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratio_range = aspect_ratio_range
        
        # Create cache directories
        self.cache_dir.mkdir(exist_ok=True)
        self.text_cache_dir.mkdir(exist_ok=True)
        
        # Process all image directories
        self.image_files = []
        self.text_files = []
        self._process_directories(image_dirs)
        
    def _process_directories(self, image_dirs: List[str]):
        """Process multiple image directories efficiently"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        
        for dir_path in image_dirs:
            dir_path = Path(dir_path)
            if not dir_path.exists():
                print(f"Warning: Directory {dir_path} does not exist, skipping")
                continue
                
            print(f"Processing directory: {dir_path}")
            
            # Find all image files with valid extensions
            for ext in image_extensions:
                image_files = list(dir_path.rglob(f"*{ext}"))
                
                # Filter images by size and aspect ratio
                for img_path in tqdm(image_files, desc=f"Validating {ext} files"):
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            
                            # Check only size constraints if aspect_ratio_range is None
                            if self.aspect_ratio_range is None:
                                size_ok = (min(width, height) >= self.min_size and
                                         max(width, height) <= self.max_size)
                            else:
                                aspect_ratio = width / height
                                size_ok = (min(width, height) >= self.min_size and
                                         max(width, height) <= self.max_size and
                                         self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1])
                            
                            if size_ok:
                                # Look for corresponding text file
                                txt_path = img_path.with_suffix('.txt')
                                if txt_path.exists():
                                    self.image_files.append(img_path)
                                    self.text_files.append(txt_path)
                                    
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue
            
            print(f"Found {len(self.image_files)} valid images in {dir_path}")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        img_path = self.image_files[idx]
        txt_path = self.text_files[idx]
        
        # Generate cache paths
        img_cache_path = self.cache_dir / f"{img_path.stem}.pt"
        text_cache_path = self.text_cache_dir / f"{img_path.stem}.pt"
        
        # Load or generate image tensor
        try:
            if img_cache_path.exists():
                img = torch.load(img_cache_path, map_location='cpu')
            else:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                torch.save(img, img_cache_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a default/empty tensor
            img = torch.zeros((3, 256, 256))
            
        # Load or generate text embeddings
        try:
            if text_cache_path.exists():
                text_data = torch.load(text_cache_path, map_location='cpu')
                text_embeds = text_data['embeds']
                tag_weight = text_data['tag_weight']
            else:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                text_embeds = self.text_embedder(caption) if hasattr(self, 'text_embedder') else {}
                tag_weight = torch.tensor(1.0)  # Default weight
                torch.save({'embeds': text_embeds, 'tag_weight': tag_weight}, text_cache_path)
        except Exception as e:
            print(f"Error loading text {txt_path}: {e}")
            # Return empty embeddings
            text_embeds = {}
            tag_weight = torch.tensor(1.0)
            
        return img, text_embeds, tag_weight
    
    def __len__(self) -> int:
        return len(self.image_files)

class NovelAIDataset(MultiDirectoryImageDataset):
    def __init__(
        self,
        image_dirs: List[str],
        transform: Optional[Callable] = None,
        device: torch.device = None,
        vae: Optional[AutoencoderKL] = None,
        cache_dir: str = "latent_cache",
        text_cache_dir: str = "text_cache"
    ):
        super().__init__(
            image_dirs=image_dirs,
            transform=transform,
            cache_dir=cache_dir,
            text_cache_dir=text_cache_dir
        )
        self.device = device
        self.vae = vae
        self.text_embedder = TextEmbedder(device=device)
        
        # Initialize optimized tag weighter
        self.tag_weighter = FastTagWeighter(
            min_weight=0.1,
            max_weight=2.0,
            default_weight=1.0,
            smoothing_factor=0.1,
            rarity_scale=0.5,
            cache_size=10000
        )
        
        # Pre-compute tag weights with optimized system
        print("Computing tag weights with optimized system...")
        self._precompute_tag_weights()
        
        # Print statistics after initialization
        self.tag_weighter.print_stats()
        
    def _precompute_tag_weights(self):
        """Pre-compute weights for all tags with optimized parsing"""
        for txt_path in tqdm(self.text_files, desc="Processing tags"):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                tags = parse_tags(caption)  # Using optimized parser
                self.tag_weighter.update_frequencies(tags)
            except Exception as e:
                print(f"Error processing tags from {txt_path}: {e}")
                continue

    def _get_default_text_embeds(self) -> Dict[str, torch.Tensor]:
        """Create default text embeddings with correct structure"""
        return {
            'base_text_embeds': torch.zeros((1, 77, 768)),  # [batch, seq_len, hidden_dim]
            'base_pooled_embeds': torch.zeros((1, 768)),    # [batch, hidden_dim]
        }

class AspectBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: NovelAIDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by latent dimensions
        self.groups = {}
        for idx in range(len(dataset)):
            img_cache_path = dataset.cache_dir / f"{dataset.image_files[idx].stem}.pt"
            if img_cache_path.exists():
                # Load cached latents to get dimensions
                try:
                    latents = torch.load(img_cache_path, map_location='cpu')
                    key = (latents.shape[2], latents.shape[3])  # Group by latent height, width
                    if key not in self.groups:
                        self.groups[key] = []
                    self.groups[key].append(idx)
                except Exception as e:
                    print(f"Error loading latents from {img_cache_path}: {e}")
                    continue
            else:
                # If latents not cached, get dimensions from original image
                try:
                    with Image.open(dataset.image_files[idx]) as img:
                        width, height = img.size
                        # Convert to latent dimensions (divide by 8)
                        latent_height = height // 8
                        latent_width = width // 8
                        key = (latent_height, latent_width)
                        if key not in self.groups:
                            self.groups[key] = []
                        self.groups[key].append(idx)
                except Exception as e:
                    print(f"Error processing {dataset.image_files[idx]}: {e}")
                    continue
        
        # Create batches from groups
        self.batches = []
        for indices in self.groups.values():
            # Create batches of exactly matching dimensions
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Only use full batches
                    self.batches.append(batch)
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)

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

class MemoryEfficientAttention(nn.Module):
    """Memory efficient attention using scaled_dot_product_attention"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Use FP8 quantization for linear layers
        self.qkv = LinearFp8(dim, dim * 3, bias=qkv_bias)
        self.proj = LinearFp8(dim, dim)
        self.dropout = dropout  # Pass dropout rate directly to scaled_dot_product_attention
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        # Project to q, k, v and split heads
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape (B, num_heads, N, head_dim)
        
        # Handle mask if provided
        if mask is not None:
            # Ensure mask has correct shape [B, 1, N] or [B, N]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            # Convert boolean mask to additive mask for scaled_dot_product_attention
            attn_mask = torch.zeros_like(mask, dtype=torch.float)
            attn_mask.masked_fill_(~mask, float('-inf'))
        else:
            attn_mask = None
            
        # Use scaled_dot_product_attention for efficient attention computation
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape and project output
        out = attn_output.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out

def replace_attention_layers(model: nn.Module):
    """Replace attention layers with memory efficient quantized version"""
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            new_attention = MemoryEfficientAttention(
                dim=module.embed_dim,
                num_heads=module.num_heads,
                qkv_bias=module.in_proj_bias is not None,
                dropout=module.dropout
            )
            setattr(model, name, new_attention)
        elif len(list(module.children())) > 0:
            replace_attention_layers(module)

class MemoryEfficientSDXLAttention(nn.Module):
    """Memory efficient attention for SDXL using scaled_dot_product_attention"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear layers without FP8 quantization for SDXL compatibility
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout  # Pass dropout rate directly to scaled_dot_product_attention
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        # Project to q, k, v and split heads
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape (B, num_heads, N, head_dim)
        
        # Handle mask if provided
        if mask is not None:
            # Ensure mask has correct shape [B, 1, N] or [B, N]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            # Convert boolean mask to additive mask for scaled_dot_product_attention
            attn_mask = torch.zeros_like(mask, dtype=torch.float)
            attn_mask.masked_fill_(~mask, float('-inf'))
        else:
            attn_mask = None
            
        # Use scaled_dot_product_attention for efficient attention computation
        # Flash Attention will be automatically used when available
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scale  # Explicitly pass scale factor
        )
        
        # Reshape and project output
        out = attn_output.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out

def replace_sdxl_attention_layers(model: nn.Module):
    """Replace SDXL attention layers with memory efficient version"""
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            new_attention = MemoryEfficientSDXLAttention(
                dim=module.embed_dim,
                num_heads=module.num_heads,
                qkv_bias=module.in_proj_bias is not None,
                dropout=module.dropout
            )
            setattr(model, name, new_attention)
        elif len(list(module.children())) > 0:
            replace_sdxl_attention_layers(module)

def is_quantized_parameter(module: nn.Module, name: str) -> bool:
    """Check if a parameter is quantized"""
    return hasattr(module, f"{name}_quantized")

def get_offload_tensor_bytes(module: nn.Module) -> int:
    """Get size of offloaded tensor in bytes"""
    total_bytes = 0
    for name, param in module.named_parameters():
        if is_quantized_parameter(module, name):
            total_bytes += param.numel() * param.element_size()
    return total_bytes

def quantize_layers(module: nn.Module, device: torch.device, train_dtype: torch.dtype):
    """Quantize layers in module"""
    for name, submodule in module.named_modules():
        if isinstance(submodule, LinearFp8):
            submodule.quantize(device)
            submodule.compute_dtype = train_dtype

def offload_quantized(module: nn.Module, device: torch.device, non_blocking: bool = True):
    """Offload quantized parameters to specified device"""
    for name, param in module.named_parameters():
        if is_quantized_parameter(module, name):
            param.data = param.data.to(device, non_blocking=non_blocking)

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

class NovelAIDiffusionV3Trainer(torch.nn.Module):
    def __init__(
        self,
        model: UNet2DConditionModel,
        vae: AutoencoderKL,
        optimizer: torch.optim.Optimizer,
        scheduler: DDPMScheduler,
        device: torch.device,
        accelerator: Optional[Accelerator] = None,
        resume_from_checkpoint: Optional[str] = None,
        max_vram_usage: float = 0.8,
        gradient_accumulation_steps: int = 4  # Add gradient accumulation steps
    ):
        super().__init__()
        
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Enable flash attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        self.model = model
        self.vae = vae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accelerator = accelerator
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Add memory manager initialization
        self.memory_manager = MemoryManager()
        
        # Initialize memory management systems
        self.layer_strategy = LayerOffloadStrategy(model, max_vram_usage)
        self.layer_conductor = LayerOffloadConductor(model, device)
        
        # Setup quantization
        self.quantization = MemoryEfficientQuantization()
        self.quantization.setup_quantization(model, device, torch.bfloat16)
        
        # Calculate total size needed for layer parameters
        total_param_size = sum(self.layer_strategy.layer_sizes.values())
        self.layer_allocator = StaticLayerAllocator(total_param_size, device)
        
        # Initialize activation management
        self.activation_allocator = StaticActivationAllocator(model)
        
        # Register all layers with conductor
        for name, module in model.named_modules():
            if len(list(module.parameters())) > 0:
                self.layer_conductor.register_layer(name, module)
        
        # Add projection layer for CLIP embeddings
        self.hidden_proj = nn.Linear(768, model.config.cross_attention_dim).to(
            device=device, 
            dtype=torch.float32
        )
        
        # Pre-allocate tensors for time embeddings
        self.register_buffer('base_area', torch.tensor(1024 * 1024, dtype=torch.float32))
        self.register_buffer('aesthetic_score', torch.tensor(6.0, dtype=torch.bfloat16))
        self.register_buffer('crop_score', torch.tensor(3.0, dtype=torch.bfloat16))
        self.register_buffer('zero_score', torch.tensor(0.0, dtype=torch.bfloat16))
        
        # Update ZTSNR parameters
        self.sigma_data = 1.0
        self.sigma_min = 0.002
        self.sigma_max = float('inf')
        self.rho = 7.0
        self.num_timesteps = 1000
        self.min_snr_gamma = 0.1
        
        # Pre-compute sigmas
        self.register_buffer('sigmas', self.get_sigmas())
        
        # Add tracking for epochs and steps
        self.current_epoch = 0
        self.global_step = 0
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
            
        # Allocate activation buffers after model is fully initialized
        self.activation_allocator.allocate_buffers(device)
        
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        added_cond_kwargs: Dict = None,
        current_layer: str = None,
        phase: str = 'forward'
    ) -> torch.Tensor:
        """Forward pass with memory-efficient layer handling"""
        # Handle layer offloading if needed
        if current_layer:
            # Get required layers for current computation
            required_layers = self.layer_strategy.get_required_layers(current_layer, phase)
            
            # Get layers to offload
            to_offload = self.layer_strategy.suggest_offload(required_layers)
            
            # Offload layers to CPU
            for layer_name in to_offload:
                self.layer_conductor.offload_to_cpu(layer_name)
                self.layer_strategy.update_vram_usage(layer_name, False)
            
            # Load required layers to GPU
            for layer_name in required_layers:
                self.layer_conductor.load_to_gpu(layer_name)
                self.layer_strategy.update_vram_usage(layer_name, True)
            
            # Synchronize transfers
            self.layer_conductor.synchronize()
        
        # Call UNet's forward pass
        return self.model(
            sample=x,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
        )[0]

    def training_step(
        self,
        latents: torch.Tensor,
        text_embeds: Dict[str, torch.Tensor],
        tag_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Memory-efficient training step with gradient accumulation"""
        # Memory management at start of step
        self.memory_manager.clear_cache()
        
        # Initialize accumulated loss
        accumulated_loss = 0
        
        for accumulation_step in range(self.gradient_accumulation_steps):
            # Get batch slice for current accumulation step
            batch_size = latents.shape[0] // self.gradient_accumulation_steps
            start_idx = accumulation_step * batch_size
            end_idx = start_idx + batch_size
            
            batch_latents = latents[start_idx:end_idx].reshape(batch_size, -1, latents.shape[-2], latents.shape[-1])
            batch_text_embeds = {k: v[start_idx:end_idx] for k, v in text_embeds.items()}
            batch_tag_weights = tag_weights[start_idx:end_idx]
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Ensure tag_weights has correct shape for broadcasting
                batch_tag_weights = batch_tag_weights.view(-1, 1, 1, 1)
                
                with torch.inference_mode(), torch.amp.autocast(dtype=torch.bfloat16):
                    # Get latent dimensions
                    height = batch_latents.shape[2] * 8
                    width = batch_latents.shape[3] * 8
                    
                    area = torch.tensor(height * width, device=self.device, dtype=torch.float32)
                    noise_scale = torch.sqrt(area / self.base_area)
                    
                    # Sample timesteps and get sigmas
                    timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device)
                    sigma = self.sigmas[timesteps] * noise_scale
                    
                    # Generate noise
                    noise = torch.randn_like(batch_latents, dtype=torch.float32, device=self.device)
                    is_infinite = torch.isinf(sigma)
                    
                    # Apply noising
                    noisy_images = torch.where(
                        is_infinite.view(-1, 1, 1, 1),
                        noise,
                        batch_latents + noise * sigma.view(-1, 1, 1, 1)
                    )
                    
                    # Get Karras scalings
                    c_skip, c_out, c_in = self.get_karras_scalings(sigma)
                    model_input = noisy_images * c_in.view(-1, 1, 1, 1)
                    
                    # Process text embeddings efficiently with non-blocking transfers
                    base_hidden = batch_text_embeds["base_text_embeds"].to(
                        device=self.device,
                        dtype=torch.bfloat16,
                        non_blocking=True
                    ).squeeze(1)
                    
                    base_pooled = batch_text_embeds["base_pooled_embeds"].to(
                        device=self.device,
                        dtype=torch.bfloat16,
                        non_blocking=True
                    ).squeeze(1)
                    
                    # Project text embeddings
                    batch_size, seq_len, _ = base_hidden.shape
                    base_hidden_float32 = base_hidden.to(dtype=torch.float32)
                    encoder_hidden_states = self.hidden_proj(
                        base_hidden_float32.view(-1, 768)
                    ).view(batch_size, seq_len, -1)
                    
                    # Get time embeddings
                    time_ids = self._get_add_time_ids(batch_latents)
                    added_cond_kwargs = {
                        "text_embeds": base_pooled,
                        "time_ids": time_ids
                    }
                
                # Forward pass with layer offloading and activation checkpointing
                v_prediction = None
                for layer_idx, (name, layer) in enumerate(self.model.named_modules()):
                    if not list(layer.parameters()):
                        continue
                    
                    if v_prediction is not None:
                        self.activation_allocator.store_activation(name, v_prediction)
                    
                    self.layer_conductor.before_layer(layer_idx, name)
                    
                    with torch.amp.autocast(dtype=torch.bfloat16):
                        v_prediction = self.forward(
                            model_input if layer_idx == 0 else v_prediction,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            added_cond_kwargs=added_cond_kwargs,
                            current_layer=name,
                            phase='forward'
                        ).sample
                    
                    self.layer_conductor.after_layer(layer_idx, name)
                    self.memory_manager.clear_cache()
                
                # Compute prediction and loss
                with torch.amp.autocast(dtype=torch.bfloat16):
                    pred_images = torch.where(
                        is_infinite.view(-1, 1, 1, 1),
                        -self.sigma_data * v_prediction,
                        noisy_images * c_skip.view(-1, 1, 1, 1) + v_prediction * c_out.view(-1, 1, 1, 1)
                    )
                    
                    # Compute loss with SNR weighting
                    loss = F.mse_loss(pred_images, batch_latents, reduction='none')
                    snr = self.get_snr(sigma)
                    min_snr = torch.tensor(self.min_snr_gamma, device=self.device)
                    snr_weights = torch.where(
                        is_infinite.view(-1, 1, 1, 1),
                        min_snr.view(-1, 1, 1, 1),
                        torch.minimum(snr, min_snr).view(-1, 1, 1, 1)
                    )
                    
                    # Apply tag weights and reduce
                    loss = loss * snr_weights * batch_tag_weights
                    loss = loss.mean() / self.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            accumulated_loss += loss.item() * self.gradient_accumulation_steps
            
            # Clear activations and cache
            self.activation_allocator.clear()
            self.memory_manager.clear_cache()
        
        # Clip gradients
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step with gradient scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return accumulated_loss, pred_images, v_prediction, timesteps

    def load_checkpoint(self, checkpoint_path: str):
        """Load model and training state from checkpoint"""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            self.current_epoch = training_state["epoch"]
            self.global_step = training_state["global_step"]
            self.optimizer.load_state_dict(training_state["optimizer_state"])
            print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        else:
            print("No training state found, starting from scratch with pretrained weights")

    @torch.no_grad()
    def _get_add_time_ids(self, images: torch.Tensor) -> torch.Tensor:
        """Optimized time_ids computation for H100"""
        batch_size = images.shape[0]
        orig_height = images.shape[2] * 8
        orig_width = images.shape[3] * 8
        
        add_time_ids = torch.empty((batch_size, 2, 4), device=self.device, dtype=torch.bfloat16)
        add_time_ids[:, 0, 0] = orig_height
        add_time_ids[:, 0, 1] = orig_width
        add_time_ids[:, 0, 2] = self.aesthetic_score
        add_time_ids[:, 0, 3] = self.zero_score
        add_time_ids[:, 1, 0] = orig_height
        add_time_ids[:, 1, 1] = orig_width
        add_time_ids[:, 1, 2] = self.crop_score
        add_time_ids[:, 1, 3] = self.zero_score
        
        return add_time_ids.reshape(batch_size, -1)

    def get_sigmas(self) -> torch.Tensor:
        """
        Generate noise schedule with zero-terminal SNR, handling infinite noise timestep.
        The last timestep is set to sigma = ∞ for ZTSNR.
        """
        # Create regular sigma schedule for t in [0,1)
        ramp = torch.linspace(0, 1, self.num_timesteps, device=self.device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        sigmas = torch.empty_like(ramp)
        
        # Regular steps (all except last)
        sigmas[:-1] = (min_inv_rho * (1 - ramp[:-1])) ** self.rho
        
        # Set final step to infinity for ZTSNR
        sigmas[-1] = float('inf')
        
        return sigmas.to(self.device)

    def get_karras_scalings(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modified Karras Preconditioner scaling factors for v-prediction with ZTSNR support.
        For σ = ∞:
            cskip(σ) = 0
            cout(σ) = -σ_data
            cin(σ) = 1/√(σ² + σ_data²)
        """
        is_infinite = torch.isinf(sigma)
        sigma_sq = sigma * sigma
        sigma_data_sq = self.sigma_data * self.sigma_data
        denominator = sigma_data_sq + sigma_sq
        denominator_sqrt = torch.sqrt(denominator)
        
        # Handle infinite sigma case explicitly
        c_skip = torch.where(is_infinite,
                            torch.zeros_like(sigma),  # cskip(∞) = 0
                            sigma_data_sq / denominator)
        
        c_out = torch.where(is_infinite,
                           -self.sigma_data * torch.ones_like(sigma),  # cout(∞) = -σ_data
                           -sigma * self.sigma_data / denominator_sqrt)
        
        c_in = 1.0 / denominator_sqrt  # cin(σ) = 1/√(σ² + σ_data²)
        
        return c_skip, c_out, c_in

    def get_snr(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Signal-to-Noise Ratio as defined in the paper for given sigma:
        SNR(σ) = (σ_data / σ)²
        """
        return (self.sigma_data / sigma)**2

    def get_minsnr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute MinSNR weights as described in the paper.
        Uses alphas from the scheduler:
        snr_t = alpha_t / (1 - alpha_t), and then clamp snr_t by min_snr_gamma.
        
        weights = min(snr_t, min_snr_gamma)
        """
        alphas = self.scheduler.alphas_cumprod.to(timesteps.device)
        alpha_t = alphas[timesteps]
        snr_t = alpha_t / (1 - alpha_t)
        min_snr = torch.tensor(self.min_snr_gamma, device=self.device)
        weights = torch.minimum(snr_t, min_snr)
        return weights.float()

    def train_epoch(self, dataloader: DataLoader, epoch: int, log_interval: int = 10) -> float:
        """Train for one epoch with performance profiling"""
        self.current_epoch = epoch
        self.model.train()
        total_loss = 0
        
        with self.profiler.start_profiling() as prof:
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                start_time = time.time()
                
                with self.profiler.profile_range("training_step"):
                    loss, _, _, _ = self.training_step(batch[0], batch[1], batch[2])
                
                # Record metrics
                batch_time = time.time() - start_time
                memory_used = torch.cuda.memory_allocated()
                self.profiler.record_step(
                    batch_time=batch_time,
                    batch_size=batch[0].shape[0],
                    memory_used=memory_used,
                    loss=loss.item()
                )
                
                # Auto-tune hyperparameters
                new_params = self.auto_tuner.update(self.profiler)
                if new_params != self.auto_tuner.current_params:
                    print("\nUpdating hyperparameters:")
                    for k, v in new_params.items():
                        print(f"  {k}: {v}")
                
                if prof is not None:
                    prof.step()
                
                total_loss += loss.item()
                
        return total_loss / len(dataloader)

    @staticmethod
    def create_dataloader(
        dataset: NovelAIDataset,
        batch_size: int,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = True
    ) -> DataLoader:
        """Create optimized dataloader for training
        
        Args:
            dataset: The NovelAIDataset instance
            batch_size: Number of samples per batch
            num_workers: Number of worker processes for data loading
            shuffle: Whether to shuffle the data
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep worker processes alive between epochs
            prefetch_factor: Number of batches to prefetch per worker
            drop_last: Drop last incomplete batch
        """
        # Create sampler that groups by latent dimensions
        sampler = AspectBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        # Create dataloader with optimized settings
        return DataLoader(
            dataset,
            batch_sampler=sampler,  # Use custom sampler for dimension-matched batches
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            generator=torch.Generator().manual_seed(42),  # Reproducible shuffling
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),  # Seed workers
        )

    def save_checkpoint(self, save_path: str):
        """Save model checkpoint with only modified components in fp16 format"""
        if self.accelerator is not None:
            # Unwrap model if using accelerator
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model

        # Create checkpoint directory
        os.makedirs(save_path, exist_ok=True)

        # Temporarily convert model to float16 for saving
        original_dtype = unwrapped_model.dtype
        unwrapped_model = unwrapped_model.to(torch.float16)

        try:
            # Save UNet weights in fp16
            unwrapped_model.save_pretrained(
                os.path.join(save_path, "unet"),
                safe_serialization=True  # Use safetensors format
            )
        finally:
            # Convert back to original dtype
            unwrapped_model = unwrapped_model.to(original_dtype)

        # Save training state
        training_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(training_state, os.path.join(save_path, "training_state.pt"))

        print(f"Saved checkpoint to {save_path} in fp16 format")

    def compute_grad_norm(self):
        """Compute total gradient norm"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def log_detailed_metrics(self,
                            loss: torch.Tensor,
                            v_pred: torch.Tensor,
                            grad_norm: float,
                            timesteps: torch.Tensor):
        """Log detailed training metrics to W&B"""
        if not self.accelerator.is_main_process:
            return
        
        # Compute v-prediction statistics
        v_pred_mean = v_pred.mean().item()
        v_pred_std = v_pred.std().item()
        v_pred_min = v_pred.min().item()
        v_pred_max = v_pred.max().item()
        
        # Log detailed metrics
        wandb.log({
            'loss/total': loss.item(),
            'v_pred/mean': v_pred_mean,
            'v_pred/std': v_pred_std,
            'v_pred/min': v_pred_min,
            'v_pred/max': v_pred_max,
            'grad/norm': grad_norm,
            'timesteps/mean': timesteps.float().mean().item(),
            'timesteps/std': timesteps.float().std().item()
        })

class MemoryManager:
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        self.oom_count = 0
        
    def update(self):
        self.current_memory = torch.cuda.memory_allocated()
        self.peak_memory = max(self.peak_memory, self.current_memory)
        
    def clear_cache(self):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch_gc()
        
    def handle_oom(self):
        self.oom_count += 1
        self.clear_cache()
        return self.oom_count > 3  # Return True if OOM is persistent
        
    def log_memory_stats(self):
        """Log current memory statistics"""
        print(f"Memory Stats:")
        print(f"  Current: {self.current_memory/1e9:.1f}GB")
        print(f"  Peak: {self.peak_memory/1e9:.1f}GB")
        print(f"  OOM Count: {self.oom_count}")

class TrainProgress:
    def __init__(
            self,
            epoch: int = 0,
            epoch_step: int = 0,
            epoch_sample: int = 0,
            global_step: int = 0,
    ):
        self.epoch = epoch
        self.epoch_step = epoch_step
        self.epoch_sample = epoch_sample
        self.global_step = global_step

    def next_step(self, batch_size: int):
        self.epoch_step += 1
        self.epoch_sample += batch_size
        self.global_step += 1

    def next_epoch(self):
        self.epoch_step = 0
        self.epoch_sample = 0
        self.epoch += 1

    def filename_string(self):
        return f"{self.global_step}-{self.epoch}-{self.epoch_step}"



class GenerateLossesModel:
    """Calculates and saves losses for each image in a dataset, sorted by loss value."""
    
    def __init__(self, output_path: str, device: torch.device = None):
        self.output_path = output_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_manager = MemoryManager()
        self.quantization = MemoryEfficientQuantization()

    def _setup_optimized_model(self, pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        """Setup model with memory and quantization optimizations"""
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name,
            subfolder="vae",
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        print("Loading UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(self.device)

        # Replace attention layers and enable optimizations
        print("Setting up optimizations...")
        replace_sdxl_attention_layers(self.unet)
        self.unet.enable_gradient_checkpointing()
        
        # Apply quantization
        self.quantization.setup_quantization(
            self.unet,
            self.device,
            torch.bfloat16
        )
        
        # Set to eval mode and disable gradients
        self.unet.eval()
        for param in self.unet.parameters():
            param.requires_grad_(False)
            
        self.memory_manager.clear_cache()

        # Log model size after quantization
        model_size = self.quantization.get_module_size(self.unet)
        print(f"Model size after quantization: {model_size/1e9:.2f}GB")

    def _process_batch_efficiently(self, batch):
        """Process batch with quantization-aware inference"""
        try:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
                # Move latents to device
                latents = batch[0].to(self.device, non_blocking=True)
                
                # Get text embeddings
                text_embeds = {k: v.to(self.device, non_blocking=True) 
                             for k, v in batch[1].items()}
                
                # Setup noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=self.device)
                noisy_latents = noise + latents
                
                # Model prediction
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeds["base_text_embeds"]
                    ).sample
                
                # Simple MSE loss
                loss = F.mse_loss(noise_pred, noise, reduction='mean')
                return float(loss)
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                if self.memory_manager.handle_oom():
                    # Try offloading to CPU
                    self.quantization.offload_to_cpu(self.unet)
                    torch.cuda.empty_cache()
                    return None
                print("Persistent OOM errors, reducing batch size")
                raise e
            raise e

    def calculate_losses(self, image_dirs: List[str]):
        """Calculate losses for all images in the directories"""
        try:
            self._setup_optimized_model()

            # Setup transform
            transform = transforms.Compose([
                transforms.ToTensor(),
                ensure_three_channels,
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                convert_to_bfloat16
            ])
            
            # Create dataset
            dataset = NovelAIDataset(
                image_dirs=image_dirs,
                transform=transform,
                device=self.device,
                vae=self.vae,
                cache_dir="latent_cache",
                text_cache_dir="text_cache"
            )
            
            # Create dataloader with batch size 1
            dataloader = NovelAIDiffusionV3Trainer.create_dataloader(
                dataset=dataset,
                batch_size=1,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )

            print("\nCalculating losses...")
            step_tqdm = tqdm(dataloader, desc="Processing images")

            filename_loss_list = []
            for idx, (img_path, _, img_cache_path, _) in enumerate(dataset.items):
                self.memory_manager.update()
                
                # Get batch from dataloader
                batch = next(iter(dataloader))
                loss = self._process_batch_efficiently(batch)
                
                if loss is not None:
                    filename_loss_list.append((img_path, loss))
                    
                    step_tqdm.set_postfix({
                        'loss': f"{loss:.4f}",
                        'memory': f"{self.memory_manager.current_memory/1e9:.1f}GB",
                        'peak': f"{self.memory_manager.peak_memory/1e9:.1f}GB"
                    })
                
                if len(filename_loss_list) % 10 == 0:
                    self.memory_manager.clear_cache()

            # Sort by loss in descending order
            filename_loss_list.sort(key=lambda x: x[1], reverse=True)
            filename_to_loss = {x[0]: x[1] for x in filename_loss_list}
            
            # Save results
            with open(self.output_path, "w") as f:
                json.dump(filename_to_loss, f, indent=4)

            print(f"\nResults saved to {self.output_path}")
            print(f"Peak memory usage: {self.memory_manager.peak_memory/1e9:.1f}GB")

        except Exception as e:
            print(f"Error during loss generation: {e}")
            raise
        finally:
            self.memory_manager.clear_cache()
            if hasattr(self, 'unet'):
                self.quantization.offload_to_cpu(self.unet)
                del self.unet
            if hasattr(self, 'vae'):
                del self.vae
            self.memory_manager.clear_cache()

def main():
    # Add argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint directory to resume from")
    parser.add_argument("--unet_path", type=str, help="Path to UNet safetensors file to start from")
    parser.add_argument("--generate_losses", type=str, help="Generate losses and save to specified JSON file")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\Datasets\collage", help="Path to dataset directory")
    args = parser.parse_args()

    # Create a variable to hold our trainer instance
    trainer = None

    def signal_handler(signum, frame):
        """Handle interrupt signals by saving checkpoint before exit"""
        signal_name = signal.Signals(signum).name
        print(f"\nReceived {signal_name} signal. Attempting to save checkpoint...")
        
        if trainer is not None:
            try:
                emergency_save_path = os.path.join("checkpoints", "emergency_checkpoint")
                trainer.save_checkpoint(emergency_save_path)
                print("Emergency checkpoint saved successfully.")
            except Exception as e:
                print(f"Failed to save emergency checkpoint: {e}")
        
        print("Exiting...")
        sys.exit(1)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # termination request

    # Basic setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # If generating losses, do that and exit
    if args.generate_losses:
        print("Generating losses for dataset...")
        loss_generator = GenerateLossesModel(args.generate_losses, device)
        loss_generator.calculate_losses([args.dataset_dir])
        return

    # Initialize wandb for training mode
    wandb.init(
        project="sdxl-finetune",
        config={
            "batch_size": 4,
            "grad_accum_steps": 4,
            "effective_batch": 16,
            "learning_rate": 4e-7,
            "num_epochs": 10,
            "model": "SDXL-base-1.0",
            "optimizer": "AdamW-BF16",
            "scheduler": "DDPM",
            "min_snr_gamma": 0.1,
            "quantization": "FP8"
        }
    )
    
    pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=torch.bfloat16
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    print("Loading UNet...")
    if args.resume_from_checkpoint:
        print(f"Loading UNet from checkpoint directory: {args.resume_from_checkpoint}")
        unet = UNet2DConditionModel.from_pretrained(
            args.resume_from_checkpoint,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)
    elif args.unet_path:
        print(f"Loading UNet from safetensors file: {args.unet_path}")
        unet_dir = os.path.dirname(args.unet_path)
        config_path = os.path.join(unet_dir, "config.json")
        
        if os.path.exists(config_path):
            print(f"Found config.json in same directory, loading from: {unet_dir}")
            unet = UNet2DConditionModel.from_pretrained(
                unet_dir,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to(device)
        else:
            print("No config.json found, loading architecture from base model")
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name,
                subfolder="unet",
                torch_dtype=torch.bfloat16,
            ).to(device)
            state_dict = load_file(args.unet_path)
            unet.load_state_dict(state_dict)
    else:
        print("Loading fresh UNet from pretrained model")
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)

    # Replace attention layers and enable optimizations
    print("Replacing attention layers with memory efficient implementation...")
    replace_sdxl_attention_layers(unet)
    unet.enable_gradient_checkpointing()
    
    # Initialize quantization
    print("Setting up quantization...")
    quantization = MemoryEfficientQuantization()
    quantization.setup_quantization(unet, device, torch.bfloat16)
    
    # Log initial model size
    model_size = quantization.get_module_size(unet)
    print(f"Model size after quantization: {model_size/1e9:.2f}GB")

    # Setup transform without lambdas
    transform = transforms.Compose([
        transforms.ToTensor(),
        ensure_three_channels,
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        convert_to_bfloat16
    ])
    
    print("Creating dataset...")
    dataset = NovelAIDataset(
        image_dirs=[
            r"/workspace/collage",
            r"/workspace/upscaled", 
            r"/workspace/High-quality-photo10k",
            r"/workspace/LAION_220k_GPT4Vision_captions",
            r"/workspace/photo-concept-bucket/train"
        ],
        transform=transform,
        device=device,
        vae=vae,
        cache_dir="latent_cache",
        text_cache_dir="text_cache"
    )
    
    print("Creating dataloader...")
    dataloader = NovelAIDiffusionV3Trainer.create_dataloader(
        dataset=dataset,
        batch_size=8,  # Increased batch size
        num_workers=min(8, os.cpu_count()),  # More workers
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    print("Setting up optimizer...")
    optimizer = AdamWBF16(
        unet.parameters(),
        lr=4e-7,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )
    
    print("Setting up noise scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name,
        subfolder="scheduler",
        torch_dtype=torch.bfloat16
    )
    
    print("Setting up accelerator...")
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir="logs",
        device_placement=True,
        cpu=False,
    )

    # Initialize tracking for TensorBoard
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="sdxl_finetune",
            config={
                "train_batch_size": 32,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 128,
                "learning_rate": 4e-7,
                "num_epochs": 10,
                "quantization": "FP8"
            }
        )
    
    # Move everything to device and prepare for accelerated training
    unet, optimizer, dataloader = accelerator.prepare(
        unet, optimizer, dataloader
    )
    
    print("Creating trainer...")
    trainer = NovelAIDiffusionV3Trainer(
        model=unet,
        vae=vae,
        optimizer=optimizer,
        scheduler=noise_scheduler,
        device=device,
        accelerator=accelerator,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Adjust starting epoch if resuming
    start_epoch = trainer.current_epoch
    print(f"Starting training from epoch {start_epoch + 1}")
    
    # Training configuration
    num_epochs = 10
    save_interval = 1000
    log_interval = 10
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nStarting training...")
    try:
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            try:
                epoch_loss = trainer.train_epoch(
                    dataloader=dataloader,
                    epoch=epoch,
                    log_interval=log_interval
                )
                
                # Log metrics including quantization stats
                if accelerator.is_main_process:
                    try:
                        model_size = trainer.quantization.get_module_size(trainer.model)
                        accelerator.log(
                            {
                                "train/loss": epoch_loss,
                                "train/epoch": epoch,
                                "train/step": trainer.global_step,
                                "memory/model_size_gb": model_size/1e9
                            },
                            step=trainer.global_step,
                        )
                    except Exception as e:
                        print(f"Error in TensorBoard logging: {e}")
                
                # Save checkpoints
                try:
                    # Save epoch checkpoint
                    epoch_checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}")
                    print(f"Saving epoch checkpoint to {epoch_checkpoint_path}")
                    trainer.save_checkpoint(epoch_checkpoint_path)
                    
                    # Save step checkpoint if interval is reached
                    if trainer.global_step % save_interval == 0:
                        step_checkpoint_path = os.path.join(
                            save_dir, 
                            f"checkpoint_step_{trainer.global_step}"
                        )
                        print(f"Saving step checkpoint to {step_checkpoint_path}")
                        trainer.save_checkpoint(step_checkpoint_path)
                        
                        # Cleanup old step checkpoints (keep last 3)
                        try:
                            step_checkpoints = sorted([
                                f for f in os.listdir(save_dir) 
                                if f.startswith("checkpoint_step_")
                            ])
                            if len(step_checkpoints) > 3:
                                for old_ckpt in step_checkpoints[:-3]:
                                    old_path = os.path.join(save_dir, old_ckpt)
                                    print(f"Removing old checkpoint: {old_path}")
                                    shutil.rmtree(old_path)
                        except Exception as e:
                            print(f"Error cleaning up old checkpoints: {e}")
                            
                except Exception as e:
                    print(f"Error saving checkpoints: {e}")
                    print(f"Current GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
                
                print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
                
            except Exception as e:
                print(f"Error in epoch {epoch+1}: {e}")
                print(f"Memory stats:")
                print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")
                print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
                raise
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...")
        final_checkpoint_path = os.path.join(save_dir, "final_checkpoint")
        trainer.save_checkpoint(final_checkpoint_path)
        print("Final checkpoint saved.")
    else:
        # Training completed normally, save final checkpoint
        print("\nTraining completed. Saving final checkpoint...")
        final_checkpoint_path = os.path.join(save_dir, "final_checkpoint")
        trainer.save_checkpoint(final_checkpoint_path)
        print("Final checkpoint saved.")
    
    # Close wandb run
    wandb.finish()
    
    # Close accelerator
    accelerator.end_training()
    print("Training completed!")

if __name__ == "__main__":
    main()