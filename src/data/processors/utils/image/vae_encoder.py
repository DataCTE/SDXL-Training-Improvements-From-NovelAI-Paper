import torch
import logging
from typing import Optional
import time
import gc
from weakref import WeakValueDictionary
from src.config.config import VAEEncoderConfig  # Import the new config class
from src.utils.logging.metrics import log_error_with_context, log_metrics, log_system_metrics
from src.data.processors.utils.batch_utils import get_gpu_memory_usage
from diffusers import AutoencoderKL
import asyncio
from collections import OrderedDict

# Add this function
def is_xformers_available():
    """Check if xformers is available."""
    try:
        import xformers
        return True
    except ImportError:
        return False

logger = logging.getLogger(__name__)

class LRUCache:
    """LRU cache with a max size."""
    def __init__(self, maxsize=128):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key not in self.cache:
            return None
        # Move to end to show recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

class VAEEncoder:
    """Handles VAE encoding with memory optimization and async support."""
    
    def __init__(self, vae: AutoencoderKL, config: VAEEncoderConfig):
        """Initialize VAE encoder with configuration."""
        self.vae = vae
        self.config = config
        
        # Freeze VAE
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        # Move to device and set dtype from config with fallback
        dtype = getattr(torch, self.config.forced_dtype, torch.float32)
        self.vae.to(device=self.config.device, dtype=dtype)
        
        # Conditionally enable slicing or xformers
        if self.config.enable_vae_slicing:
            self.vae.enable_slicing()
        
        if self.config.enable_xformers_attention and is_xformers_available():
            self.vae.enable_xformers_memory_efficient_attention()
        
        logging.debug(
            f"Initialized VAEEncoder with device={self.config.device}, dtype={dtype}, "
            f"slicing={self.config.enable_vae_slicing}, xformers={self.config.enable_xformers_attention}"
        )

        # Add tensor cache for frequently used sizes
        self._tensor_cache = WeakValueDictionary()
        
        # Add batch processing queue
        self.batch_queue_size = getattr(config, 'batch_queue_size', 32)
        self.processing_queue = []
        
        # Enable memory optimizations
        self._enable_optimizations()
        
        # Enable flash attention if available
        self._setup_flash_attention()
        
        # Setup tensor parallel processing
        self._setup_tensor_parallel()
        
        # Initialize prefetch queue for latent caching
        self.prefetch_queue = asyncio.Queue(maxsize=64)
        self.latent_cache = LRUCache(maxsize=1024)  # Add import for LRU cache
        
        # Initialize memory pool for efficient allocation
        self.memory_pool = torch.cuda.graph.CUDAGraph() if torch.cuda.is_available() else None

    def _enable_optimizations(self):
        """Enable all available optimizations for faster processing."""
        if torch.cuda.is_available():
            # Enable TF32 for faster processing on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set optimal cudnn algorithms
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        # Always enable VAE slicing for memory efficiency
        self.vae.enable_slicing()
        
        if self.config.enable_xformers_attention and is_xformers_available():
            self.vae.enable_xformers_memory_efficient_attention()

    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clear any tensor cache if you plan to use caching
            # if hasattr(self, '_tensor_cache'):
            #     self._tensor_cache.clear()

            # Clear CUDA cache if using GPU
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()

            # Clear references to the VAE
            del self.vae
            gc.collect()

            logger.info("Successfully cleaned up VAE encoder resources.")

        except Exception as e:
            logger.error(f"Error during VAE encoder cleanup: {str(e)}")
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except:
                pass

    def __del__(self):
        """Cleanup when encoder is deleted."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"Error during VAE encoder destructor: {e}")

    def _enable_memory_efficient_attention(self) -> None:
        """Enable memory efficient attention if possible."""
        try:
            import xformers
            self.vae.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention for VAE")
        except ImportError:
            logger.warning("xformers not available, using standard attention")

    # Optional caching function (commented out usage in encode_images)
    # def _get_cached_tensor(self, key: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    #     """Get or create tensor from cache."""
    #     tensor = self._tensor_cache.get(key)
    #     if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
    #         tensor = torch.empty(shape, dtype=dtype, device=self.config.device)
    #         self._tensor_cache[key] = tensor
    #     return tensor

    @torch.inference_mode()
    async def encode_images(self, images: "torch.Tensor", keep_on_gpu: bool = False) -> "torch.Tensor":
        """Optimized batch encoding with dynamic batching and prefetching."""
        try:
            # Check cache first
            cache_key = self._get_cache_key(images)
            if cached_latents := self.latent_cache.get(cache_key):
                return cached_latents

            # Use CUDA graphs for repeated operations
            if self.memory_pool and images.shape in self._static_shapes:
                return self._encode_with_cuda_graph(images, keep_on_gpu)

            # Optimize memory allocation
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use more GPU memory
            
            # Process in optimal chunks with dynamic batching
            chunks = self._get_optimal_chunks(images)
            latents = await self._process_chunks_parallel(chunks, keep_on_gpu)
            
            # Cache results
            self.latent_cache[cache_key] = latents
            
            # Prefetch next likely encodings
            asyncio.create_task(self._prefetch_next_batch(images))
            
            return latents

        except Exception as e:
            logger.error(f"VAE encoding error: {str(e)}")
            return None

    async def _process_chunks_parallel(self, chunks, keep_on_gpu):
        """Process chunks in parallel with advanced optimizations."""
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self._process_chunk(chunk, keep_on_gpu))
                for chunk in chunks
            ]
        
        results = [t.result() for t in tasks]
        return self._combine_results(results)

    @torch.compile(fullgraph=True, dynamic=False)
    def _process_chunk(self, chunk, keep_on_gpu):
        """Optimized chunk processing with TorchDynamo."""
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            return self.vae.encode(chunk).latent_dist.sample()

    def _get_optimal_chunks(self, images):
        """Get optimal chunk sizes based on hardware and model."""
        if torch.cuda.is_available():
            # Use hardware-specific optimizations
            sm_count = torch.cuda.get_device_properties(0).multi_processor_count
            return self._split_for_hardware(images, sm_count)
        return [images]

    def _split_for_hardware(self, images, sm_count):
        """Split batch optimally for hardware."""
        chunk_size = max(1, (sm_count * 128) // images.shape[-1])
        return torch.chunk(images, chunk_size)

    def _setup_flash_attention(self):
        """Enable flash attention for faster processing."""
        try:
            from flash_attn import flash_attn_func
            self.use_flash_attention = True
            # Patch VAE attention layers to use flash attention
            for module in self.vae.modules():
                if isinstance(module, torch.nn.MultiheadAttention):
                    module._flash_attention_func = flash_attn_func
        except ImportError:
            self.use_flash_attention = False

    def _setup_tensor_parallel(self):
        """Setup tensor parallelism for multi-GPU processing."""
        if torch.cuda.device_count() > 1:
            self.vae = torch.nn.parallel.DistributedDataParallel(
                self.vae,
                device_ids=[self.config.device],
                output_device=self.config.device,
                find_unused_parameters=True
            )

    def _get_optimal_batch_size(self, images: torch.Tensor) -> int:
        """Determine optimal batch size based on available memory and image size."""
        try:
            if not torch.cuda.is_available():
                return self.config.vae_batch_size

            total_memory = torch.cuda.get_device_properties(self.vae.device).total_memory
            free_memory = torch.cuda.memory_reserved(self.vae.device)
            used_memory = torch.cuda.memory_allocated(self.vae.device)
            
            # Calculate actual available memory
            available_memory = total_memory - used_memory
            safe_memory = int(available_memory * 0.8)  # Leave 20% buffer
            
            # Calculate memory per image
            sample_latent = self.vae.encode(images[0:1]).latent_dist.sample()
            bytes_per_image = sample_latent.element_size() * sample_latent.nelement()
            
            optimal_size = max(1, safe_memory // bytes_per_image)
            return min(optimal_size, self.config.vae_batch_size)

        except Exception as e:
            logger.warning(f"Error calculating optimal batch size: {e}")
            return self.config.vae_batch_size

    def _get_cached_tensor(self, key: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Get or create tensor from cache."""
        tensor = self._tensor_cache.get(key)
        if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
            tensor = torch.empty(shape, dtype=dtype, device=self.vae.device)
            self._tensor_cache[key] = tensor
        return tensor

    def _get_latent_shape(self, images: torch.Tensor) -> tuple:
        """Calculate latent shape without encoding."""
        b, c, h, w = images.shape
        return (b, 4, h // 8, w // 8)
