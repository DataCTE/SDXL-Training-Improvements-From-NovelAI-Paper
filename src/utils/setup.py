import torch
import logging
import traceback
from torch.utils.data import DataLoader
from bitsandbytes.optim import AdamW8bit
from transformers.optimization import Adafactor
from diffusers.optimization import get_scheduler
from data.dataset import CustomDataset
from data.tag_weighter import TagBasedLossWeighter
from training.ema import EMAModel
from diffusers import StableDiffusionXLPipeline
from inference.text_to_image import SDXLInference
from training.vae_finetuner import VAEFineTuner
from diffusers import EulerDiscreteScheduler
import os

logger = logging.getLogger(__name__)

def setup_logging():
    """Configure basic logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def setup_torch_backends():
    """Configure PyTorch backend settings for optimal performance"""
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for a model"""
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        logger.warning(f"Model {type(model).__name__} doesn't support gradient checkpointing")

def setup_models(args, device, dtype):
    """Initialize and configure all models"""
    logger.info("Starting model setup process...")
    
    try:
        models = {}
        
        # Step 1: Load complete SDXL pipeline
        logger.info("Step 1/2: Loading SDXL pipeline...")
        try:
            # Load pipeline from local path or Hugging Face hub
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.model_path,
                torch_dtype=dtype,
                use_safetensors=True,
                dtype=dtype
            )
            
            # Add architecture verification
            logger.info("Verifying model architecture...")
            expected_keys = set(pipeline.unet.state_dict().keys())
            
            # Check if model_path is a local directory or HF model ID
            if os.path.isdir(args.model_path):
                # Local path
                weights_path = os.path.join(args.model_path, "unet/diffusion_pytorch_model.safetensors")
            else:
                # For HF models, we'll compare against the loaded state dict
                loaded_keys = expected_keys
                logger.info("Using Hugging Face model - skipping local file verification")
            
            if 'loaded_keys' in locals():
                missing_keys = expected_keys - loaded_keys
                unexpected_keys = loaded_keys - expected_keys
                
                if missing_keys:
                    logger.warning(f"Missing keys in checkpoint: {len(missing_keys)} keys")
                    logger.debug(f"First few missing keys: {list(missing_keys)[:5]}")
                
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                    logger.debug(f"First few unexpected keys: {list(unexpected_keys)[:5]}")
            
            pipeline.to(device)
            
            # Extract components
            models["unet"] = pipeline.unet
            models["vae"] = pipeline.vae
            models["text_encoder"] = pipeline.text_encoder
            models["text_encoder_2"] = pipeline.text_encoder_2
            models["tokenizer"] = pipeline.tokenizer
            models["tokenizer_2"] = pipeline.tokenizer_2
            
            # Set VAE to eval mode
            models["vae"].requires_grad_(False)
            models["vae"].eval()
            
            logger.info("SDXL pipeline loaded and components extracted successfully")
            
        except Exception as e:
            logger.error("Failed to load SDXL pipeline")
            error_msg = clean_error_message(traceback.format_exc())
            logger.error(error_msg)
            
            # Add detailed error analysis
            if "size mismatch" in str(e):
                analyze_size_mismatches(str(e))
            raise
        
        # Step 2: Initialize EMA
        logger.info("Step 2/2: Setting up EMA...")
        if args.use_ema:
            try:
                logger.info("  Initializing EMA model with parameters:")
                logger.info(f"  - Base decay rate: {args.ema_decay}")
                logger.info(f"  - Update after step: {args.ema_update_after_step}")
                logger.info(f"  - Update frequency: {args.ema_update_every}")
                logger.info(f"  - Warmup enabled: {args.use_ema_warmup}")
                
                ema_model = EMAModel(
                    model=models["unet"],
                    decay=args.ema_decay,
                    update_after_step=args.ema_update_after_step,
                    inv_gamma=args.ema_inv_gamma,
                    power=args.ema_power,
                    min_decay=args.ema_min_decay,
                    max_decay=args.ema_max_decay,
                    device=device,
                    update_every=args.ema_update_every,
                    use_ema_warmup=args.use_ema_warmup,
                    grad_scale_factor=args.ema_grad_scale_factor
                )
                logger.info("  EMA model initialized successfully")
                models["ema_model"] = ema_model
            except Exception as e:
                logger.error("  Failed to initialize EMA model")
                error_msg = clean_error_message(traceback.format_exc())
                logger.error(error_msg)
                raise
        else:
            logger.info("  Skipping EMA initialization (not enabled)")
            models["ema_model"] = None
        
        # Final verification
        logger.info("Verifying model initialization...")
        required_models = ["unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
        missing_models = [model for model in required_models if model not in models]
        
        if missing_models:
            raise ValueError(f"Missing required models: {', '.join(missing_models)}")
        
        logger.info(" All models initialized successfully")
        logger.info("Model setup completed successfully")
        return models
        
    except Exception as e:
        error_msg = clean_error_message(traceback.format_exc())
        logger.error("Model setup failed")
        logger.error(error_msg)
        raise

def analyze_size_mismatches(error_msg):
    """Analyze and log detailed information about size mismatches"""
    logger.error("\nDetailed size mismatch analysis:")
    
    # Extract all size mismatch information
    mismatches = []
    for line in error_msg.split('\n'):
        if "size mismatch for" in line:
            param = line.split('size mismatch for ')[1].split(':')[0]
            shapes = line.split(': ')[1].split('copying a param with shape ')[1]
            expected = shapes.split(', the shape in current model is ')[0]
            actual = shapes.split(', the shape in current model is ')[1]
            mismatches.append({
                'param': param,
                'expected': expected,
                'actual': actual
            })
    
    # Group mismatches by pattern
    pattern_groups = {}
    for mismatch in mismatches:
        param_parts = mismatch['param'].split('.')
        pattern = '.'.join(param_parts[:-2])  # Group by module path
        if pattern not in pattern_groups:
            pattern_groups[pattern] = []
        pattern_groups[pattern].append(mismatch)
    
    # Log analysis
    logger.error(f"Found {len(mismatches)} size mismatches in {len(pattern_groups)} module groups:")
    for pattern, group in pattern_groups.items():
        logger.error(f"\nModule: {pattern}")
        logger.error(f"Number of mismatches: {len(group)}")
        logger.error("Example mismatches:")
        for mismatch in group[:3]:  # Show first 3 examples
            logger.error(f"  Parameter: {mismatch['param']}")
            logger.error(f"  Expected: {mismatch['expected']}")
            logger.error(f"  Actual: {mismatch['actual']}")

def setup_training(args, models, device, dtype):
    """Setup training components"""
    logger.info("Starting training setup process...")
    
    try:
        components = {}
        
        # Step 1: Validate and setup basic parameters
        logger.info("Step 1/7: Validating training parameters...")
        try:
            if not hasattr(args, 'num_workers'):
                args.num_workers = min(8, torch.get_num_threads() or 1)
                logger.info(f"  Using default num_workers: {args.num_workers}")
            logger.info("✓ Training parameters validated")
        except Exception as e:
            logger.error("✗ Failed to validate training parameters")
            error_msg = clean_error_message(traceback.format_exc())
            logger.error(error_msg)
            raise

        # Step 2: Initialize Dataset
        logger.info("Step 2/7: Initializing dataset...")
        try:
            dataset_config = {
                "data_dir": args.data_dir,
                "vae": models["vae"],
                "tokenizer": models["tokenizer"],
                "tokenizer_2": models["tokenizer_2"],
                "text_encoder": models["text_encoder"],
                "text_encoder_2": models["text_encoder_2"],
                "cache_dir": args.cache_dir,
                "no_caching_latents": args.no_caching_latents,
                "all_ar": args.all_ar,
                "num_workers": args.num_workers
            }
            logger.info(f"  Dataset configuration:")
            logger.info(f"  - Data directory: {args.data_dir}")
            logger.info(f"  - Cache directory: {args.cache_dir}")
            logger.info(f"  - All AR mode: {args.all_ar}")
            logger.info(f"  - Num workers: {args.num_workers}")
            
            dataset = CustomDataset(**dataset_config)
            components["dataset"] = dataset
            logger.info("✓ Dataset initialized successfully")
        except Exception as e:
            logger.error("✗ Failed to initialize dataset")
            error_msg = clean_error_message(traceback.format_exc())
            logger.error(error_msg)
            raise

        # Step 3: Setup DataLoader
        logger.info("Step 3/7: Setting up data loader...")
        try:
            train_dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=dataset.collate_fn,
                num_workers=args.num_workers,
                pin_memory=True
            )
            components["train_dataloader"] = train_dataloader
            logger.info(f"  DataLoader configured with batch size: {args.batch_size}")
            logger.info("✓ DataLoader setup complete")
        except Exception as e:
            logger.error("✗ Failed to setup data loader")
            error_msg = clean_error_message(traceback.format_exc())
            logger.error(error_msg)
            raise

        # Step 4: Initialize Optimizer and Scheduler
        logger.info("Step 4/7: Initializing optimizer and scheduler...")
        try:
            if args.use_adafactor:
                logger.info("  Using Adafactor optimizer")
                optimizer = Adafactor(
                    models["unet"].parameters(),
                    lr=args.learning_rate * args.batch_size,
                    scale_parameter=True,
                    relative_step=False,
                    warmup_init=False
                )
            else:
                logger.info("  Using AdamW8bit optimizer")
                optimizer = AdamW8bit(
                    models["unet"].parameters(),
                    lr=args.learning_rate * args.batch_size,
                    betas=(0.9, 0.999)
                )
            
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_training_steps = args.num_epochs * num_update_steps_per_epoch
            
            logger.info("  Configuring learning rate scheduler...")
            lr_scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=num_training_steps
            )
            
            components.update({
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "num_update_steps_per_epoch": num_update_steps_per_epoch,
                "num_training_steps": num_training_steps
            })
            logger.info(f"  Total training steps: {num_training_steps}")
            logger.info("✓ Optimizer and scheduler initialized")
        except Exception as e:
            logger.error("✗ Failed to initialize optimizer and scheduler")
            error_msg = clean_error_message(traceback.format_exc())
            logger.error(error_msg)
            raise

        # Step 5: Initialize VAE Finetuner (if enabled)
        logger.info("Step 5/7: Setting up VAE finetuner...")
        try:
            if hasattr(args, 'vae_finetuning') and args.vae_finetuning:
                logger.info("  Initializing VAE finetuner with parameters:")
                vae_config = {
                    'learning_rate': args.vae_learning_rate,
                    'min_snr_gamma': args.min_snr_gamma,
                    'adaptive_loss_scale': args.adaptive_loss_scale,
                    'kl_weight': args.kl_weight,
                    'perceptual_weight': args.perceptual_weight,
                    'use_8bit_adam': args.use_8bit_adam,
                    'gradient_checkpointing': args.gradient_checkpointing,
                    'mixed_precision': args.mixed_precision,
                    'use_channel_scaling': args.use_channel_scaling
                }
                for key, value in vae_config.items():
                    logger.info(f"  - {key}: {value}")
                
                vae_finetuner = VAEFineTuner(models["vae"], **vae_config)
                components["vae_finetuner"] = vae_finetuner
                logger.info("✓ VAE finetuner initialized")
            else:
                logger.info("  Skipping VAE finetuner (not enabled)")
                components["vae_finetuner"] = None
        except Exception as e:
            logger.error("✗ Failed to initialize VAE finetuner")
            error_msg = clean_error_message(traceback.format_exc())
            logger.error(error_msg)
            raise

        # Step 6: Initialize Additional Components
        logger.info("Step 6/7: Initializing additional components...")
        try:
            # Tag weighter
            logger.info("  Setting up tag-based loss weighter...")
            tag_weighter = TagBasedLossWeighter(
                min_weight=args.min_tag_weight,
                max_weight=args.max_tag_weight
            )
            components["tag_weighter"] = tag_weighter
            
            # Noise scheduler
            logger.info("  Configuring noise scheduler...")
            noise_scheduler = EulerDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                use_karras_sigmas=True,
                sigma_min=args.sigma_min,
                sigma_max=160.0,
                steps_offset=1,
            )
            logger.info("✓ Additional components initialized")
        except Exception as e:
            logger.error("✗ Failed to initialize additional components")
            error_msg = clean_error_message(traceback.format_exc())
            logger.error(error_msg)
            raise

        # Step 7: Setup Validator
        logger.info("Step 7/7: Setting up validator...")
        try:
            validator = SDXLInference(
                model_path=args.model_path,
                device=device,
                dtype=dtype,
                use_resolution_binning=True
            )

            validator.pipeline = StableDiffusionXLPipeline(
                vae=models["vae"],
                text_encoder=models["text_encoder"],
                text_encoder_2=models["text_encoder_2"],
                tokenizer=models["tokenizer"],
                tokenizer_2=models["tokenizer_2"],
                unet=models["unet"],
                scheduler=noise_scheduler,
            ).to(device)
            
            components["validator"] = validator
            logger.info("✓ Validator setup complete")
        except Exception as e:
            logger.error("✗ Failed to setup validator")
            error_msg = clean_error_message(traceback.format_exc())
            logger.error(error_msg)
            raise

        # Final verification
        logger.info("Verifying training components...")
        required_components = ["dataset", "train_dataloader", "optimizer", "lr_scheduler", "tag_weighter", "validator"]
        missing_components = [comp for comp in required_components if comp not in components]
        
        if missing_components:
            raise ValueError(f"Missing required components: {', '.join(missing_components)}")
        
        logger.info("✓ All training components initialized successfully")
        logger.info("Training setup completed successfully")
        return components
        
    except Exception as e:
        error_msg = clean_error_message(traceback.format_exc())
        logger.error("✗ Training setup failed")
        logger.error(error_msg)
        raise

def clean_error_message(error_msg):
    """Clean up error messages and identify key mismatch locations"""
    lines = error_msg.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if "size mismatch for" in line:
            try:
                # Extract parameter name and location
                frame = next((l for l in lines if 'File "' in l and '.py", line' in l), None)
                if frame:
                    file_path = frame.split('File "')[1].split('", line')[0]
                    line_num = frame.split('", line ')[1].split(',')[0]
                    param = line.split('size mismatch for ')[1].split(':')[0]
                    shapes = line.split(': ')[1].split('copying a param with shape ')[1]
                    expected = shapes.split(', the shape in current model is ')[0]
                    actual = shapes.split(', the shape in current model is ')[1]
                    
                    cleaned_lines.append(f"Shape mismatch in {file_path}:{line_num}")
                    cleaned_lines.append(f"Parameter: {param}")
                    cleaned_lines.append(f"Expected shape: {expected}")
                    cleaned_lines.append(f"Actual shape: {actual}")
            except:
                cleaned_lines.append(line)
        elif 'File "' in line and '.py", line' in line:
            cleaned_lines.append(line)
        elif any(x in line.lower() for x in ['error', 'exception', 'failed', 'traceback']):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)