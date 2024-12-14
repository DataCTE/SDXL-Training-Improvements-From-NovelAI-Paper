import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from accelerate import Accelerator
from adamw_bf16 import AdamWBF16
import os
import wandb
import argparse
from safetensors.torch import load_file
import signal
import torch.distributed as dist
from collections import defaultdict
from accelerate import Accelerator
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear8bitLt, LinearNf4
    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False
    Linear8bitLt = None
    LinearNf4 = None
    bnb = None

try:
    from torch.cuda import empty_cache as torch_gc
except ImportError:
    def torch_gc():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# Import all the necessary components from modules
from data.dataset import NovelAIDataset
from data.sampler import AspectBatchSampler
from trainers.sdxl_trainer import NovelAIDiffusionV3Trainer
from configs.training_config import TrainingConfig
from utils.distributed import setup_distributed, cleanup_distributed
from utils.reporting import setup_wandb

def load_unet(args, config: TrainingConfig, device: torch.device):
    """Load UNet model based on arguments"""
    if args.resume_from_checkpoint:
        print(f"Loading UNet from checkpoint directory: {args.resume_from_checkpoint}")
        return UNet2DConditionModel.from_pretrained(
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
                config.pretrained_model_name,
                subfolder="unet",
                torch_dtype=torch.bfloat16,
            ).to(device)
            state_dict = load_file(args.unet_path)
            unet.load_state_dict(state_dict)
        return unet
    else:
        print("Loading fresh UNet from pretrained model")
        return UNet2DConditionModel.from_pretrained(
            config.pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)

def signal_handler(signum, frame):
    global trainer
    
    if dist.is_initialized():
        shutdown_tensor = torch.tensor(1, device="cuda")
        dist.all_reduce(shutdown_tensor)
        dist.barrier()
    
    print(f"\nReceived {signal.Signals(signum).name} signal. Attempting to save checkpoint...")
    if trainer is not None and (not dist.is_initialized() or dist.get_rank() == 0):
        try:
            emergency_save_path = os.path.join("checkpoints", "emergency_checkpoint")
            trainer.save_checkpoint(emergency_save_path)
            print("Emergency checkpoint saved successfully.")
        except Exception as e:
            print(f"Failed to save emergency checkpoint: {e}")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def setup_training_optimizations():
    # Enable tensor cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set optimal CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    
    # Configure memory allocator
    torch.cuda.memory.set_per_process_memory_fraction(0.95)
    torch.cuda.memory.set_per_process_memory_fraction(0.95)

def optimize_cuda_allocator():
    """Configure CUDA memory allocator for optimal performance"""
    # Use cudnn autotuner
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory allocator settings
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.memory_stats()
    torch.cuda.empty_cache()
    
    # Enable flash attention if available
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)

def main():
    global trainer
    trainer = None
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--unet_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str)
    parser.add_argument("--max_vram_usage", type=float, default=0.8)
    args = parser.parse_args()
    
    # Setup distributed training first
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = setup_distributed(args.local_rank, args.world_size)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        config = TrainingConfig()
        
        # Initialize wandb only on main process
        if args.local_rank == -1 or args.local_rank == 0:
            setup_wandb(config)
        
        # Set seeds for reproducibility
        seed = config.seed if hasattr(config, 'seed') else 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
        # Load models
        unet = load_unet(args, config, device)
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae",
            torch_dtype=torch.bfloat16
        ).to(device)
      
        # Initialize optimizer with memory-efficient settings
        if HAVE_BNB:
            optimizer = bnb.optim.AdamW8bit(
                unet.parameters(),
                lr=config.learning_rate,
                betas=config.adam_betas,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = AdamWBF16(
                unet.parameters(),
                lr=config.learning_rate,
                betas=config.adam_betas,
                weight_decay=config.weight_decay,
            )
        
        # Initialize dataset with distributed info
        dataset = NovelAIDataset(
            image_dirs=config.image_dirs,
            transform=config.get_transform(),
            device=device,
            vae=vae,
            cache_dir=config.cache_dir,
            text_cache_dir=config.text_cache_dir,
            #local_rank=args.local_rank,
            #world_size=args.world_size if args.local_rank != -1 else 1
        )
        
        sampler = AspectBatchSampler(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            #world_size=args.world_size if args.local_rank != -1 else 1,
            #rank=args.local_rank
        )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Initialize accelerator if using distributed training
        if args.local_rank != -1:
            accelerator = Accelerator(
                gradient_accumulation_steps=config.grad_accum_steps,
                mixed_precision="bf16",
                log_with="wandb",
                project_dir="logs"
            )
        else:
            accelerator = None
        
        # Initialize trainer with memory management and config
        trainer = NovelAIDiffusionV3Trainer(
            model=unet,
            vae=vae,
            optimizer=optimizer,
            scheduler=DDPMScheduler(),
            device=device,
            batch_size=config.batch_size,
            accelerator=accelerator,
            resume_from_checkpoint=args.resume_from_checkpoint,
            max_vram_usage=args.max_vram_usage,
            gradient_accumulation_steps=config.grad_accum_steps,
            config=config
        )
        
        if args.resume_from_checkpoint:
            if args.local_rank != -1:
                torch.distributed.barrier()
            if args.local_rank == 0:
                trainer.load_checkpoint(args.resume_from_checkpoint)
            if args.local_rank != -1:
                torch.distributed.barrier()
        
        # Broadcast model state to all processes
        if args.local_rank != -1:
            for param in unet.parameters():
                dist.broadcast(param.data, src=0)
            for state in optimizer.state.values():
                for tensor_name, tensor_value in state.items():
                    if isinstance(tensor_value, torch.Tensor):
                        dist.broadcast(tensor_value, src=0)
        
        # Training loop with improved memory management
        for epoch in range(trainer.current_epoch, config.num_epochs):
            if args.local_rank != -1:
                sampler.set_epoch(epoch)
            trainer.current_epoch = epoch
            
            # Clear cache before each epoch
            torch_gc()
            
            avg_loss = trainer.train_epoch(dataloader, epoch)
            
            if args.local_rank == -1 or args.local_rank == 0:
                print(f"Epoch {epoch} completed with average loss: {avg_loss:.4f}")
                trainer.save_checkpoint()
                
    finally:
        cleanup_distributed()
        if args.local_rank == -1 or args.local_rank == 0:
            wandb.finish()

if __name__ == "__main__":
    main()

