import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import os

logger = logging.getLogger(__name__)

class LatentCacheManager:
    """Manages caching and retrieval of VAE latents."""
    
    def __init__(self, cache_dir: str = "latents_cache", vae: Optional[torch.nn.Module] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.vae = vae
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.vae is not None:
            self.device = next(self.vae.parameters()).device
        
        # In-memory cache
        self.latents_cache: Dict[str, torch.Tensor] = {}
        
        # Worker pool for processing
        self.num_workers = 1  # Single worker for GPU operations
        self.process_pool = ThreadPoolExecutor(max_workers=self.num_workers)
    
    def process_single_latent(self, image_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Process a single image into latent space."""
        try:
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    image_tensor = image_tensor.to(self.device)
                    latent = self.vae.encode(image_tensor).latent_dist.sample()
                    latent = latent.cpu()  # Move back to CPU for storage
                    
            # Clean up GPU memory
            del image_tensor
            torch.cuda.empty_cache()
                
            return latent
            
        except Exception as e:
            logger.error(f"Failed to process latent: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
    
    def process_latents_batch(self, image_tensors: torch.Tensor) -> Optional[torch.Tensor]:
        """Process a batch of images into latent space efficiently."""
        try:
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    image_tensors = image_tensors.to(self.device)
                    latents = self.vae.encode(image_tensors).latent_dist.sample()
                    latents = latents.cpu()  # Move back to CPU for storage
                    
            # Clean up GPU memory
            del image_tensors
            torch.cuda.empty_cache()
                
            return latents
            
        except Exception as e:
            logger.error(f"Failed to process latent batch: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
            
    def get_latents(self, image_path: str, image_tensor: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Get latents from cache or compute them."""
        # Check memory cache first
        if image_path in self.latents_cache:
            return self.latents_cache[image_path]
            
        # Check disk cache
        latent_path = self.cache_dir / f"{hash(image_path)}.pt"
        if latent_path.exists():
            try:
                return torch.load(latent_path)
            except Exception as e:
                logger.error(f"Failed to load cached latent for {image_path}: {str(e)}")
        
        # Compute latents if we have the VAE and image tensor
        if self.vae is not None and image_tensor is not None:
            latents = self.process_single_latent(image_tensor)
            if latents is not None:
                # Cache in memory
                self.latents_cache[image_path] = latents
                # Cache to disk
                try:
                    torch.save(latents, latent_path)
                except Exception as e:
                    logger.error(f"Failed to cache latent for {image_path}: {str(e)}")
                return latents
        
        return None
    
    def initialize_workers(self) -> None:
        """Initialize worker processes for parallel latent computation"""
        if self.vae is not None:
            self.process_pool = torch.multiprocessing.Pool(processes=torch.multiprocessing.cpu_count())
            logger.info("Initialized latent processing worker pool")
            
    def close_workers(self) -> None:
        """Close worker processes"""
        if hasattr(self, 'process_pool'):
            self.process_pool.close()
            self.process_pool.join()
            logger.info("Closed latent processing worker pool")
    
    def offload_to_disk(self):
        """Offload in-memory cache to disk."""
        try:
            for path, latent in self.latents_cache.items():
                latent_path = self.cache_dir / f"{hash(path)}.pt"
                torch.save(latent, latent_path)
            
            # Clear memory cache
            self.latents_cache.clear()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to offload latents to disk: {str(e)}")
            raise
