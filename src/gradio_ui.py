import gradio as gr
import yaml
import os
from typing import List
import torch
from pathlib import Path
import wandb
import plotly.graph_objects as go
from datetime import datetime
import json
from train import main as train_main
from src.config.config import Config

def create_training_ui():
    # Store training metrics
    class TrainingState:
        def __init__(self):
            self.losses = []
            self.learning_rates = []
            self.steps = []
            self.current_epoch = 0
            self.wandb_run = None
    
    state = TrainingState()

    def update_training_plots(losses, learning_rates, steps):
        # Create loss plot
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=steps, y=losses, name='Training Loss'))
        loss_fig.update_layout(title='Training Loss Over Time',
                             xaxis_title='Steps',
                             yaxis_title='Loss')

        # Create learning rate plot
        lr_fig = go.Figure()
        lr_fig.add_trace(go.Scatter(x=steps, y=learning_rates, name='Learning Rate'))
        lr_fig.update_layout(title='Learning Rate Schedule',
                           xaxis_title='Steps',
                           yaxis_title='Learning Rate')

        return loss_fig, lr_fig

    def update_config(
        pretrained_model: str,
        image_dirs: str,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        gradient_accumulation_steps: int,
        mixed_precision: str,
        num_workers: int,
        wandb_project: str,
        wandb_entity: str,
        wandb_tags: str,
        log_interval: int,
        save_interval: int,
        resume_checkpoint: str = None,
        unet_path: str = None,
        enable_wandb: bool = True,
        # Distributed training options
        distributed_training: bool = True,
        backend: str = "nccl",
        use_fsdp: bool = True,
        cpu_offload: bool = False,
        full_shard: bool = True,
        sync_batch_norm: bool = True,
        min_num_params_per_shard: int = int(1e6),
    ):
        # Create experiment name
        experiment_name = f"sdxl-train-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create base config structure
        config = {
            "model": {
                "pretrained_model_name": pretrained_model
            },
            "data": {
                "image_dirs": image_dirs.split(","),
                "cache_dir": "cache",
                "text_cache_dir": "text_cache",
                "num_workers": num_workers,
                "pin_memory": True,
                "persistent_workers": True
            },
            "training": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "mixed_precision": mixed_precision,
                "optimizer_betas": [0.9, 0.999],
                "weight_decay": 1e-2,
                "optimizer_eps": 1e-8,
                "log_steps": log_interval,
                "save_steps": save_interval
            },
            "paths": {
                "checkpoints_dir": f"checkpoints/{experiment_name}",
                "logs_dir": f"logs/{experiment_name}",
                "output_dir": f"outputs/{experiment_name}"
            },
            "logging": {
                "wandb_project": wandb_project,
                "wandb_entity": wandb_entity,
                "wandb_tags": [tag.strip() for tag in wandb_tags.split(",") if tag.strip()],
                "experiment_name": experiment_name
            },
            "system": {
                "enable_xformers": True,
                "channels_last": True,
                "gradient_checkpointing": True,
                "cudnn_benchmark": True,
                "disable_debug_apis": True,
                "mixed_precision": mixed_precision,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                # Distributed training settings
                "distributed_training": distributed_training,
                "backend": backend,
                "use_fsdp": use_fsdp,
                "cpu_offload": cpu_offload,
                "full_shard": full_shard,
                "sync_batch_norm": sync_batch_norm,
                "min_num_params_per_shard": min_num_params_per_shard,
                "forward_prefetch": True,
                "backward_prefetch": True,
                "limit_all_gathers": True,
                "find_unused_parameters": False
            }
        }

        # Save config to yaml file
        config_path = f"configs/{experiment_name}.yaml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Initialize WandB if enabled
        if enable_wandb:
            state.wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=experiment_name,
                config=config,
                tags=config["logging"]["wandb_tags"],
                resume="allow"
            )

        # Prepare training arguments
        train_args = ["--config", config_path]
        if resume_checkpoint:
            train_args.extend(["--resume_from_checkpoint", resume_checkpoint])
        if unet_path:
            train_args.extend(["--unet_path", unet_path])

        # Start training
        try:
            import sys
            sys.argv[1:] = train_args
            train_main()
            return "Training completed successfully!", None, None
        except Exception as e:
            if state.wandb_run:
                state.wandb_run.finish()
            return f"Training failed with error: {str(e)}", None, None

    # Create the Gradio interface
    with gr.Blocks(title="SDXL Training Interface") as interface:
        gr.Markdown("# SDXL Training Interface")
        
        with gr.Tab("Training Configuration"):
            with gr.Row():
                pretrained_model = gr.Textbox(
                    label="Pretrained Model Name",
                    value="stabilityai/stable-diffusion-xl-base-1.0"
                )
                image_dirs = gr.Textbox(
                    label="Image Directories (comma-separated)",
                    placeholder="/path/to/images1,/path/to/images2"
                )
            
            with gr.Row():
                batch_size = gr.Number(label="Batch Size", value=1)
                learning_rate = gr.Number(label="Learning Rate", value=1e-5)
                num_epochs = gr.Number(label="Number of Epochs", value=100)
                
            with gr.Row():
                gradient_accumulation = gr.Number(
                    label="Gradient Accumulation Steps",
                    value=4
                )
                mixed_precision = gr.Dropdown(
                    label="Mixed Precision",
                    choices=["no", "fp16", "bf16"],
                    value="bf16"
                )
                num_workers = gr.Number(label="Number of Workers", value=4)
                
            with gr.Row():
                log_interval = gr.Number(label="Log Interval", value=10)
                save_interval = gr.Number(label="Save Interval", value=1000)
                
            with gr.Row():
                resume_checkpoint = gr.Textbox(
                    label="Resume from Checkpoint (optional)",
                    placeholder="/path/to/checkpoint"
                )
                unet_path = gr.Textbox(
                    label="UNet Path (optional)",
                    placeholder="/path/to/unet.safetensors"
                )
                
            with gr.Row():
                wandb_project = gr.Textbox(label="W&B Project", value="sdxl-finetune")
                wandb_entity = gr.Textbox(label="W&B Entity")
                wandb_tags = gr.Textbox(
                    label="W&B Tags (comma-separated)",
                    placeholder="tag1,tag2,tag3"
                )
                enable_wandb = gr.Checkbox(label="Enable W&B Logging", value=True)
                
            # Add distributed training options
            with gr.Tab("Distributed Training"):
                with gr.Row():
                    distributed_training = gr.Checkbox(
                        label="Enable Distributed Training",
                        value=True
                    )
                    backend = gr.Dropdown(
                        label="Backend",
                        choices=["nccl", "gloo"],
                        value="nccl"
                    )
                    
                with gr.Row():
                    use_fsdp = gr.Checkbox(
                        label="Use FSDP",
                        value=True,
                        info="Use Fully Sharded Data Parallel instead of DDP"
                    )
                    cpu_offload = gr.Checkbox(
                        label="CPU Offload",
                        value=False,
                        info="Offload parameters to CPU to save GPU memory"
                    )
                    
                with gr.Row():
                    full_shard = gr.Checkbox(
                        label="Full Sharding",
                        value=True,
                        info="Use full parameter sharding strategy"
                    )
                    sync_batch_norm = gr.Checkbox(
                        label="Sync BatchNorm",
                        value=True,
                        info="Synchronize BatchNorm across devices"
                    )
                    
                min_num_params = gr.Number(
                    label="Min Params per Shard",
                    value=1e6,
                    info="Minimum number of parameters per GPU for FSDP"
                )

        with gr.Tab("Training Metrics"):
            with gr.Row():
                loss_plot = gr.Plot(label="Training Loss")
                lr_plot = gr.Plot(label="Learning Rate")
            
            with gr.Row():
                current_loss = gr.Number(label="Current Loss")
                current_lr = gr.Number(label="Current Learning Rate")
                current_step = gr.Number(label="Current Step")

        start_btn = gr.Button("Start Training")
        output = gr.Textbox(label="Training Status")

        start_btn.click(
            fn=update_config,
            inputs=[
                pretrained_model,
                image_dirs,
                batch_size,
                learning_rate,
                num_epochs,
                gradient_accumulation,
                mixed_precision,
                num_workers,
                wandb_project,
                wandb_entity,
                wandb_tags,
                log_interval,
                save_interval,
                resume_checkpoint,
                unet_path,
                enable_wandb,
                # Add distributed training inputs
                distributed_training,
                backend,
                use_fsdp,
                cpu_offload,
                full_shard,
                sync_batch_norm,
                min_num_params
            ],
            outputs=[output, loss_plot, lr_plot]
        )

    return interface

if __name__ == "__main__":
    interface = create_training_ui()
    interface.launch(share=True) 