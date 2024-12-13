import wandb
from typing import Optional, Dict, Any
from dataclasses import asdict

def setup_wandb(config: Any) -> None:
    """Initialize Weights & Biases logging with configuration"""
    
    # Convert config to dict if it's a dataclass
    if hasattr(config, '__dataclass_fields__'):
        config_dict = asdict(config)
    else:
        config_dict = vars(config)
    
    # Initialize wandb with project configuration
    wandb.init(
        project="sdxl-finetune",
        config={
            "batch_size": config_dict.get("batch_size", 4),
            "grad_accum_steps": config_dict.get("gradient_accumulation_steps", 4),
            "effective_batch": config_dict.get("batch_size", 4) * config_dict.get("gradient_accumulation_steps", 4),
            "learning_rate": config_dict.get("learning_rate", 4e-7),
            "num_epochs": config_dict.get("num_epochs", 10),
            "model": "SDXL-base-1.0",
            "optimizer": "AdamW-BF16",
            "scheduler": "DDPM",
            "min_snr_gamma": config_dict.get("min_snr_gamma", 0.1),
            "quantization": "FP8",
            **{k:v for k,v in config_dict.items() if k not in ["batch_size", "gradient_accumulation_steps", "learning_rate", "num_epochs", "min_snr_gamma"]}
        }
    )

def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log training metrics to WandB"""
    wandb.log(metrics, step=step)

def log_model_artifact(model_path: str, name: str, type: str = "model") -> None:
    """Log model checkpoint as WandB artifact"""
    artifact = wandb.Artifact(name=name, type=type)
    artifact.add_dir(model_path)
    wandb.log_artifact(artifact)

def finish_logging() -> None:
    """Cleanup WandB logging"""
    wandb.finish()