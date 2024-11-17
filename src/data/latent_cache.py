"""
High-performance latent caching system for SDXL training.

This module provides efficient caching of VAE latents using multi-GPU processing
with optimized memory management and throughput.

Classes:
    GPUWorker: Handles VAE encoding on individual GPU devices
    LatentCacheManager: Manages the distributed caching system
"""

import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Set, Any
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
import torch.multiprocessing as mp
from queue import Empty
from threading import Lock

# Set the start method to 'spawn' for CUDA multiprocessing
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

logger = logging.getLogger(__name__)

class GPUWorker:
    """Worker class for processing VAE encoding on a specific GPU.
    
    This class handles the encoding of image batches to latent space using
    a VAE model on a dedicated GPU device. It implements efficient memory
    management and batch processing optimizations.
    
    Attributes:
        gpu_id: ID of the GPU device to use
        worker_id: Unique identifier for this worker
        device: PyTorch device object
        vae: VAE model for encoding
        queue_in: Input queue for receiving batches
        queue_out: Output queue for sending results
    """
    
    def __init__(self, gpu_id: int, worker_id: int, vae: Any,
                 queue_in: mp.Queue, queue_out: mp.Queue) -> None:
        """Initialize the GPU worker.
        
        Args:
            gpu_id: GPU device ID to use
            worker_id: Unique identifier for this worker
            vae: VAE model for encoding
            queue_in: Input queue for receiving batches
            queue_out: Output queue for sending results
        """
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.vae = vae.to(self.device)
        self.queue_in = queue_in
        self.queue_out = queue_out
        
        # Enable cudnn benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True
        logger.info("Initialized worker %d on GPU %d", worker_id, gpu_id)
        
    def process_batch(self) -> None:
        """Process batches of images from the input queue.
        
        Continuously processes batches from the input queue until a stop signal
        is received. Handles memory management and error recovery.
        """
        while True:
            try:
                batch_data = self.queue_in.get(timeout=5)
                if batch_data is None:  # Stop signal
                    logger.info(
                        "Worker %d on GPU %d received stop signal",
                        self.worker_id, self.gpu_id
                    )
                    break
                    
                batch_tensor, batch_paths = batch_data
                
                # Pre-allocate CUDA memory
                if not isinstance(batch_tensor, torch.Tensor):
                    batch_tensor = torch.stack(batch_tensor)
                
                # Move to GPU efficiently with non-blocking transfer
                batch_tensor = batch_tensor.to(
                    self.device, non_blocking=True
                )
                
                # Process in mixed precision for speed
                with torch.amp.autocast('cuda', enabled=True):
                    with torch.no_grad():
                        # Process in chunks to avoid OOM
                        chunk_size = 32  # Optimal batch size
                        all_latents = []
                        
                        for i in range(0, len(batch_tensor), chunk_size):
                            chunk = batch_tensor[i:i + chunk_size]
                            latents_chunk = self.vae.encode(chunk).latent_dist.sample()
                            all_latents.append(latents_chunk.cpu())
                            
                        latents = torch.cat(all_latents, dim=0)
                
                # Clean up GPU memory efficiently
                del batch_tensor
                del all_latents
                torch.cuda.empty_cache()
                
                # Send results back
                self.queue_out.put((batch_paths, latents))
                
            except Empty:
                continue
            except Exception as error:
                logger.error(
                    "GPU %d worker %d error: %s",
                    self.gpu_id, self.worker_id, str(error)
                )
                self.queue_out.put((None, None))
                continue

class LatentCacheManager:
    """Manages distributed caching and retrieval of VAE latents.
    
    This class coordinates multiple GPU workers to efficiently process and cache
    VAE latents. It implements optimized memory management, prefetching, and
    thread-safe caching mechanisms.
    
    Attributes:
        cache_dir: Directory for storing cached latents
        vae: VAE model for encoding
        num_gpus: Number of available GPU devices
        workers_per_gpu: Number of workers per GPU device
        total_workers: Total number of active workers
        queue_size: Size of processing queues
        prefetch_size: Size of prefetch queue
        latents_cache: In-memory cache of latents
        cached_files: Set of cached file hashes
        queue_in: Input queue for processing
        queue_out: Output queue for results
        prefetch_queue: Queue for prefetching batches
        workers: List of active worker processes
    """
    
    def __init__(
        self,
        cache_dir: str = "latents_cache",
        vae: Optional[torch.nn.Module] = None,
        workers_per_gpu: int = 4
    ) -> None:
        """Initialize the latent cache manager.
        
        Args:
            cache_dir: Directory for storing cached latents
            vae: VAE model for encoding
            workers_per_gpu: Number of workers per GPU device
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.vae = vae
        
        # Configure GPU resources
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            self.device = torch.device("cpu")
            self.num_gpus = 1
            workers_per_gpu = 1
            
        self.workers_per_gpu = workers_per_gpu
        self.total_workers = self.num_gpus * workers_per_gpu
        
        # Configure queue sizes for optimal throughput
        self.queue_size = 100
        logger.info(
            "Initializing %d workers across %d GPUs (%d per GPU)",
            self.total_workers, self.num_gpus, workers_per_gpu
        )
        
        # Initialize thread-safe caching
        self.latents_cache: Dict[str, torch.Tensor] = {}
        self.cache_lock = Lock()
        
        # Initialize cached files tracking
        self.cached_files: Set[str] = set()
        self._load_cached_files()
        
        # Initialize processing queues
        self.queue_in = mp.Queue(maxsize=self.queue_size)
        self.queue_out = mp.Queue(maxsize=self.queue_size)
        self.workers: List[mp.Process] = []
        
        # Initialize prefetching
        self.prefetch_size = 3
        self.prefetch_queue = mp.Queue(maxsize=self.prefetch_size)

        # Initialize progress tracking
        self.total_images = 0
        self.processed_images = 0
        self.pbar = None

    def _load_cached_files(self) -> None:
        """Load list of already cached files."""
        self.cached_files = {f.stem for f in self.cache_dir.glob("*.pt")}
        logger.info("Found %d existing cached latents", len(self.cached_files))

    def _get_cache_path(self, image_path: str) -> Path:
        """Get the cache file path for an image."""
        return self.cache_dir / f"{hash(image_path)}.pt"

    def is_cached(self, image_path: str) -> bool:
        """Check if an image's latents are already cached."""
        return str(hash(image_path)) in self.cached_files

    def process_latents_batch(
        self, image_tensors: torch.Tensor, batch_paths: List[str]
    ) -> None:
        """Queue a batch of images for processing with prefetching."""
        # Split into smaller batches for better memory management
        batch_size = 32  # Optimal batch size
        for i in range(0, len(image_tensors), batch_size):
            batch_tensor = image_tensors[i:i + batch_size]
            batch_path = batch_paths[i:i + batch_size]
            self.queue_in.put((batch_tensor, batch_path))

    def get_latents(
        self, image_path: str, image_tensor: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Get latents from cache or compute them."""
        # Check memory cache first
        if image_path in self.latents_cache:
            return self.latents_cache[image_path]
            
        # Check disk cache
        cache_path = self._get_cache_path(image_path)
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as error:
                logger.error(
                    "Failed to load cached latent for %s: %s",
                    image_path, str(error)
                )
        
        # Compute latents if we have the VAE and image tensor
        if self.vae is not None and image_tensor is not None:
            # Process single image on first available GPU
            device = torch.device(f'cuda:{0}')
            # Update to use newer torch.amp.autocast API
            with torch.amp.autocast('cuda', enabled=True):
                with torch.no_grad():
                    image_tensor = image_tensor.to(device)
                    latents = self.vae.encode(image_tensor).latent_dist.sample()
                    latents = latents.cpu()
            
            if latents is not None:
                # Cache in memory
                with self.cache_lock:
                    self.latents_cache[image_path] = latents
                # Cache to disk
                try:
                    torch.save(latents, cache_path)
                    self.cached_files.add(str(hash(image_path)))
                except Exception as error:
                    logger.error(
                        "Failed to cache latent for %s: %s",
                        image_path, str(error)
                    )
                return latents
        
        return None

    def initialize_latent_caching(self, image_paths: List[str]) -> bool:
        """Initialize progress bar for latent caching."""
        # Count uncached images
        uncached_images = [path for path in image_paths if not self.is_cached(path)]
        total_new = len(uncached_images)
        
        if total_new == 0:
            logger.info("All images are already cached, skipping latent generation")
            return False
            
        logger.info("Found %d uncached images to process", total_new)
        self.total_images = total_new
        self.processed_images = 0
        self.pbar = tqdm(total=total_new, desc="Caching latents", unit="img")
        return True
    
    def initialize_workers(self) -> None:
        """Initialize GPU workers for parallel latent computation"""
        if self.vae is not None:
            logger.info(
                "Initializing %d workers (%d per GPU)",
                self.total_workers, self.workers_per_gpu
            )
            worker_id = 0
            for gpu_id in range(self.num_gpus):
                for _ in range(self.workers_per_gpu):
                    worker = mp.Process(
                        target=GPUWorker(gpu_id, worker_id, self.vae, self.queue_in, self.queue_out).process_batch
                    )
                    worker.start()
                    self.workers.append(worker)
                    worker_id += 1
    
    def process_results(self) -> None:
        """Process results from GPU workers."""
        try:
            batch_paths, latents = self.queue_out.get(timeout=5)
            if batch_paths is None:  # Error occurred
                return
                
            # Cache results
            for idx, path in enumerate(batch_paths):
                with self.cache_lock:
                    self.latents_cache[path] = latents[idx]
                cache_path = self._get_cache_path(path)
                torch.save(latents[idx], cache_path)
                self.cached_files.add(str(hash(path)))
            
            # Update progress
            if self.pbar is not None:
                self.processed_images += len(batch_paths)
                self.pbar.update(len(batch_paths))
                
        except Empty:
            pass
            
    def close_workers(self) -> None:
        """Close worker processes"""
        # Send stop signals to all workers
        for _ in range(self.total_workers):
            self.queue_in.put(None)
            
        # Wait for workers to finish
        for worker in self.workers:
            worker.join()
            
        # Clear queues
        while not self.queue_in.empty():
            self.queue_in.get()
        while not self.queue_out.empty():
            self.queue_out.get()
            
        logger.info("Closed %d GPU workers", len(self.workers))
        
        if self.pbar is not None:
            self.pbar.close()
    
    def offload_to_disk(self) -> None:
        """Offload in-memory cache to disk."""
        try:
            with self.cache_lock:
                for path, latent in self.latents_cache.items():
                    cache_path = self._get_cache_path(path)
                    torch.save(latent, cache_path)
                    self.cached_files.add(str(hash(path)))
                
                # Clear memory cache
                self.latents_cache.clear()
                torch.cuda.empty_cache()
            
        except Exception as error:
            logger.error("Failed to offload latents to disk: %s", str(error))
            raise
