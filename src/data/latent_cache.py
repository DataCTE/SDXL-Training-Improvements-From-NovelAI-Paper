import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Set
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
    def __init__(self, gpu_id: int, worker_id: int, vae, queue_in, queue_out):
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.vae = vae.to(self.device)
        self.queue_in = queue_in
        self.queue_out = queue_out
        logger.info(f"Initialized worker {worker_id} on GPU {gpu_id}")
        
    def process_batch(self):
        while True:
            try:
                batch_data = self.queue_in.get(timeout=5)
                if batch_data is None:  # Stop signal
                    logger.info(f"Worker {self.worker_id} on GPU {self.gpu_id} received stop signal")
                    break
                    
                batch_tensor, batch_paths = batch_data
                
                # Update to use newer torch.amp.autocast API
                with torch.amp.autocast('cuda', enabled=True):
                    with torch.no_grad():
                        batch_tensor = batch_tensor.to(self.device)
                        latents = self.vae.encode(batch_tensor).latent_dist.sample()
                        latents = latents.cpu()
                
                # Clean up GPU memory
                del batch_tensor
                torch.cuda.empty_cache()
                
                # Send results back
                self.queue_out.put((batch_paths, latents))
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"GPU {self.gpu_id} worker {self.worker_id} error: {str(e)}")
                self.queue_out.put((None, None))
                continue

class LatentCacheManager:
    """Manages caching and retrieval of VAE latents."""
    
    def __init__(self, cache_dir: str = "latents_cache", vae: Optional[torch.nn.Module] = None, workers_per_gpu: int = 4):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.vae = vae
        
        # Get available GPUs
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            self.device = torch.device("cpu")
            self.num_gpus = 1  
            workers_per_gpu = 1
            
        self.workers_per_gpu = workers_per_gpu
        self.total_workers = self.num_gpus * workers_per_gpu
        logger.info(f"Initializing {self.total_workers} workers across {self.num_gpus} GPUs ({workers_per_gpu} per GPU)")
        
        # In-memory cache with thread safety
        self.latents_cache: Dict[str, torch.Tensor] = {}
        self.cache_lock = Lock()
        
        # Track cached files
        self.cached_files: Set[str] = set()
        self._load_cached_files()
        
        # Multiprocessing queues
        self.queue_in = mp.Queue()
        self.queue_out = mp.Queue()
        self.workers = []

        # Progress tracking
        self.total_images = 0
        self.processed_images = 0
        self.pbar = None
    
    def _load_cached_files(self):
        """Load list of already cached files."""
        self.cached_files = {f.stem for f in self.cache_dir.glob("*.pt")}
        logger.info(f"Found {len(self.cached_files)} existing cached latents")
    
    def _get_cache_path(self, image_path: str) -> Path:
        """Get the cache file path for an image."""
        return self.cache_dir / f"{hash(image_path)}.pt"
    
    def is_cached(self, image_path: str) -> bool:
        """Check if an image's latents are already cached."""
        return str(hash(image_path)) in self.cached_files
    
    def process_latents_batch(self, image_tensors: torch.Tensor, batch_paths: List[str]) -> None:
        """Queue a batch of images for processing."""
        self.queue_in.put((image_tensors, batch_paths))
    
    def get_latents(self, image_path: str, image_tensor: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Get latents from cache or compute them."""
        # Check memory cache first
        if image_path in self.latents_cache:
            return self.latents_cache[image_path]
            
        # Check disk cache
        cache_path = self._get_cache_path(image_path)
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                logger.error(f"Failed to load cached latent for {image_path}: {str(e)}")
        
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
                except Exception as e:
                    logger.error(f"Failed to cache latent for {image_path}: {str(e)}")
                return latents
        
        return None

    def initialize_latent_caching(self, image_paths: List[str]):
        """Initialize progress bar for latent caching."""
        # Count uncached images
        uncached_images = [path for path in image_paths if not self.is_cached(path)]
        total_new = len(uncached_images)
        
        if total_new == 0:
            logger.info("All images are already cached, skipping latent generation")
            return False
            
        logger.info(f"Found {total_new} uncached images to process")
        self.total_images = total_new
        self.processed_images = 0
        self.pbar = tqdm(total=total_new, desc="Caching latents", unit="img")
        return True
    
    def initialize_workers(self) -> None:
        """Initialize GPU workers for parallel latent computation"""
        if self.vae is not None:
            logger.info(f"Initializing {self.total_workers} workers ({self.workers_per_gpu} per GPU)")
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
            
        logger.info(f"Closed {len(self.workers)} GPU workers")
        
        if self.pbar is not None:
            self.pbar.close()
    
    def offload_to_disk(self):
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
            
        except Exception as e:
            logger.error(f"Failed to offload latents to disk: {str(e)}")
            raise
