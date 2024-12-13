import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchvision import transforms
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from memory.EfficientQuantization import MemoryEfficientQuantization
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