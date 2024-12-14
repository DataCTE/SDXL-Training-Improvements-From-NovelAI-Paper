import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchvision import transforms
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from memory.EfficientQuantization import EfficientQuantization
from memory.EfficientAttention import replace_sdxl_attention_layers
from memory.layeroffloading import ensure_three_channels, convert_to_bfloat16
from memory.Manager import MemoryManager
from data.dataset import NovelAIDataset
from trainers.sdxl_trainer import NovelAIDiffusionV3Trainer
from tqdm import tqdm
import json

class GenerateLossesModel:
    """Calculates and saves losses for each image in a dataset, sorted by loss value."""
    
    def __init__(self, output_path: str, device: torch.device = None):
        self.output_path = output_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_manager = MemoryManager()
        self.quantization = EfficientQuantization()

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
        """Process batch with optimized inference and memory handling"""
        try:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                # Prefetch next batch to GPU asynchronously
                latents = batch[0].to(self.device, non_blocking=True)
                text_embeds = {k: v.to(self.device, non_blocking=True) 
                             for k, v in batch[1].items()}
                
                # Generate noise in fp16 for better performance
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    noise = torch.randn_like(latents, device=self.device)
                    timesteps = torch.randint(0, 1000, (latents.shape[0],), device=self.device)
                
                # Fuse noise addition for better memory efficiency
                noisy_latents = torch.addcmul(latents, noise, torch.ones_like(latents))
                
                # Run model prediction with memory-efficient attention
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeds["base_text_embeds"],
                    return_dict=False
                )[0]  # Avoid dict overhead
                
                # Compute loss efficiently
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='mean')
                return float(loss)
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                if not self.memory_manager.handle_oom():
                    # Try emergency memory recovery
                    torch.cuda.empty_cache()
                    self.quantization.offload_to_cpu(self.unet, non_blocking=True)
                    return None
                print("Attempting recovery from OOM")
                return None
            raise e

    def calculate_losses(self, image_dirs: List[str]):
        """Calculate losses with improved batching and caching"""
        try:
            self._setup_optimized_model()

            # Optimize transform pipeline
            transform = transforms.Compose([
                transforms.ToTensor(),
                ensure_three_channels,
                transforms.Normalize([0.5], [0.5]),  # Single value normalization is faster
                convert_to_bfloat16
            ])
            
            # Create dataset with optimized caching
            dataset = NovelAIDataset(
                image_dirs=image_dirs,
                transform=transform,
                device=self.device,
                vae=self.vae,
                cache_dir="latent_cache",
                text_cache_dir="text_cache",
                prefetch_factor=2  # Enable prefetching
            )
            
            # Create optimized dataloader
            dataloader = NovelAIDiffusionV3Trainer.create_dataloader(
                dataset=dataset,
                batch_size=1,
                num_workers=min(8, len(image_dirs)),  # Scale workers with dataset
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )

            print("\nCalculating losses...")
            step_tqdm = tqdm(dataloader, desc="Processing images")

            # Pre-allocate list for better memory efficiency
            filename_loss_list = []
            
            # Process in micro-batches for better GPU utilization
            micro_batch_size = 4
            current_batch = []
            
            for idx, (img_path, _, img_cache_path, _) in enumerate(dataset.items):
                self.memory_manager.update()
                
                # Get batch and accumulate
                batch = next(iter(dataloader))
                current_batch.append((img_path, batch))
                
                # Process micro-batch
                if len(current_batch) >= micro_batch_size:
                    self._process_micro_batch(current_batch, filename_loss_list, step_tqdm)
                    current_batch = []
                    
                if len(filename_loss_list) % 20 == 0:
                    self.memory_manager.clear_cache()
            
            # Process remaining items
            if current_batch:
                self._process_micro_batch(current_batch, filename_loss_list, step_tqdm)

            # Sort and save results efficiently
            self._save_results(filename_loss_list)

        except Exception as e:
            print(f"Error during loss generation: {e}")
            raise
        finally:
            self._cleanup()
            
    def _process_micro_batch(self, batch_items, filename_loss_list, progress_bar):
        """Process a micro-batch of items efficiently"""
        try:
            # Combine batches for parallel processing
            paths = [item[0] for item in batch_items]
            batches = [item[1] for item in batch_items]
            
            # Stack inputs for batch processing
            combined_latents = torch.cat([b[0] for b in batches])
            combined_embeds = {
                k: torch.cat([b[1][k] for b in batches])
                for k in batches[0][1].keys()
            }
            
            # Process combined batch
            loss = self._process_batch_efficiently((combined_latents, combined_embeds))
            
            if loss is not None:
                # Split loss for individual items
                avg_loss = loss / len(batch_items)
                for path in paths:
                    filename_loss_list.append((path, avg_loss))
                
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'memory': f"{self.memory_manager.current_memory/1e9:.1f}GB",
                    'batch': len(batch_items)
                })
                
        except Exception as e:
            print(f"Error in micro-batch processing: {e}")
            
    def _save_results(self, filename_loss_list):
        """Save results efficiently"""
        filename_loss_list.sort(key=lambda x: x[1], reverse=True)
        
        # Use more efficient JSON writing
        with open(self.output_path, "w", buffering=1024*1024) as f:  # 1MB buffer
            f.write('{\n')
            for i, (filename, loss) in enumerate(filename_loss_list):
                f.write(f'    "{filename}": {loss:.6f}')
                if i < len(filename_loss_list) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            f.write('}\n')

        print(f"\nResults saved to {self.output_path}")
        print(f"Peak memory usage: {self.memory_manager.peak_memory/1e9:.1f}GB")
            
    def _cleanup(self):
        """Cleanup resources efficiently"""
        self.memory_manager.clear_cache()
        if hasattr(self, 'unet'):
            self.quantization.offload_to_cpu(self.unet, non_blocking=True)
            del self.unet
        if hasattr(self, 'vae'):
            del self.vae
        torch.cuda.empty_cache()