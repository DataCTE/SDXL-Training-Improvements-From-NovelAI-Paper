"""
VAE-specific caching functionality for SDXL training.

This module provides specialized caching for VAE latents using multi-GPU processing
with optimized memory management and throughput.
"""

import logging
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Any
from queue import Empty
from tqdm import tqdm

from .memory import MemoryCache

# Set the start method to 'spawn' for CUDA multiprocessing
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

logger = logging.getLogger(__name__)

class GPUWorker:
    """Worker class for processing VAE encoding on a specific GPU."""
    
    def __init__(self, gpu_id: int, worker_id: int, vae: Any,
                 queue_in: mp.Queue, queue_out: mp.Queue) -> None:
        """Initialize the GPU worker."""
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
        """Process batches from the input queue."""
        while True:
            try:
                batch_data = self.queue_in.get(timeout=5)
                if batch_data is None:  # Stop signal
                    break
                    
                batch_tensor, batch_paths = batch_data
                
                if not isinstance(batch_tensor, torch.Tensor):
                    batch_tensor = torch.stack(batch_tensor)
                
                batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                
                with torch.amp.autocast('cuda', enabled=True):
                    with torch.no_grad():
                        chunk_size = 64
                        all_latents = []
                        
                        for i in range(0, len(batch_tensor), chunk_size):
                            chunk = batch_tensor[i:i + chunk_size]
                            latents_chunk = self.vae.encode(chunk).latent_dist.sample()
                            all_latents.append(latents_chunk.cpu())
                            
                        latents = torch.cat(all_latents, dim=0)
                
                del batch_tensor
                del all_latents
                torch.cuda.empty_cache()
                
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

class VAECache:
    """Manages distributed caching of VAE latents."""
    
    def __init__(
        self,
        cache_dir: str = "vae_cache",
        vae: Optional[torch.nn.Module] = None,
        workers_per_gpu: int = 20
    ) -> None:
        """Initialize the VAE cache."""
        self.memory_cache = MemoryCache(cache_dir)
        self.vae = vae
        
        # Configure GPU resources
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            self.device = torch.device("cpu")
            self.num_gpus = 1
            workers_per_gpu = 1
            
        self.workers_per_gpu = workers_per_gpu
        self.total_workers = self.num_gpus * workers_per_gpu
        
        # Configure queues
        self.queue_size = max(200, self.total_workers * 4)
        self.queue_in = mp.Queue(maxsize=self.queue_size)
        self.queue_out = mp.Queue(maxsize=self.queue_size)
        self.workers: List[mp.Process] = []
        
    def get_latents(
        self, image_path: str, image_tensor: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Get VAE latents from cache or compute them."""
        # Try to get from cache first
        latents = self.memory_cache.get(image_path)
        if latents is not None:
            return latents
            
        # Compute latents if we have the VAE and image tensor
        if self.vae is not None and image_tensor is not None:
            device = torch.device(f'cuda:{0}')
            with torch.amp.autocast('cuda', enabled=True):
                with torch.no_grad():
                    image_tensor = image_tensor.to(device)
                    latents = self.vae.encode(image_tensor).latent_dist.sample()
                    latents = latents.cpu()
            
            if latents is not None:
                self.memory_cache.put(image_path, latents)
                return latents
        
        return None
        
    def process_batch(
        self, image_tensors: torch.Tensor, batch_paths: List[str]
    ) -> None:
        """Queue a batch for processing."""
        batch_size = 64
        for i in range(0, len(image_tensors), batch_size):
            batch_tensor = image_tensors[i:i + batch_size]
            batch_path = batch_paths[i:i + batch_size]
            self.queue_in.put((batch_tensor, batch_path))
            
    def initialize_workers(self) -> None:
        """Initialize GPU workers for parallel computation."""
        if self.vae is not None:
            worker_id = 0
            for gpu_id in range(self.num_gpus):
                for _ in range(self.workers_per_gpu):
                    worker = mp.Process(
                        target=GPUWorker(
                            gpu_id, worker_id, self.vae,
                            self.queue_in, self.queue_out
                        ).process_batch
                    )
                    worker.start()
                    self.workers.append(worker)
                    worker_id += 1
                    
    def process_results(self, pbar: Optional[tqdm] = None) -> None:
        """Process results from GPU workers."""
        try:
            batch_paths, latents = self.queue_out.get(timeout=5)
            if batch_paths is None:
                return
                
            for idx, path in enumerate(batch_paths):
                self.memory_cache.put(path, latents[idx])
            
            if pbar is not None:
                pbar.update(len(batch_paths))
                
        except Empty:
            pass
            
    def close_workers(self) -> None:
        """Close worker processes."""
        for _ in range(self.total_workers):
            self.queue_in.put(None)
            
        for worker in self.workers:
            worker.join()
            
        while not self.queue_in.empty():
            self.queue_in.get()
        while not self.queue_out.empty():
            self.queue_out.get()
            
    def offload_to_disk(self) -> None:
        """Offload cache to disk."""
        self.memory_cache.offload_to_disk()