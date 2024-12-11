import os
import torch
import shutil
from pathlib import Path
from utils.error_handling import error_handler

class CheckpointManager:
    def __init__(self, save_dir: str = "checkpoints", max_checkpoints: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
    @error_handler
    def save_checkpoint(self, model, optimizer, epoch, global_step, name=None):
        """Save a checkpoint with specified name"""
        if name is None:
            name = f"checkpoint_step_{global_step}"
            
        save_path = self.save_dir / name
        save_path.mkdir(exist_ok=True)
        
        # Save model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(
                save_path / "unet",
                safe_serialization=True
            )
            
        # Save training state
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(training_state, save_path / "training_state.pt")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self):
        """Keep only the most recent checkpoints"""
        checkpoints = sorted([
            f for f in os.listdir(self.save_dir)
            if f.startswith("checkpoint_step_")
        ])
        
        if len(checkpoints) > self.max_checkpoints:
            for old_ckpt in checkpoints[:-self.max_checkpoints]:
                old_path = self.save_dir / old_ckpt
                shutil.rmtree(old_path)
                
    @error_handler
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint and return the training state"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
            
        training_state_path = checkpoint_path / "training_state.pt"
        if not training_state_path.exists():
            raise ValueError(f"No training state found at {training_state_path}")
            
        return torch.load(training_state_path) 