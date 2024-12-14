from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class TrainProgress:
    """Enhanced progress tracking for training"""
    epoch: int = 0
    epoch_step: int = 0
    epoch_sample: int = 0
    global_step: int = 0
    start_time: float = 0.0
    samples_per_second: float = 0.0
    last_checkpoint_step: int = 0
    best_loss: Optional[float] = None
    
    def __post_init__(self):
        self.start_time = time.time()
    
    def next_step(self, batch_size: int):
        """Update progress for next step with timing metrics"""
        self.epoch_step += 1
        self.epoch_sample += batch_size
        self.global_step += 1
        
        # Calculate throughput
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.samples_per_second = self.epoch_sample / elapsed
    
    def next_epoch(self):
        """Reset epoch counters and preserve global progress"""
        self.epoch_step = 0
        self.epoch_sample = 0
        self.epoch += 1
        self.start_time = time.time()
    
    def update_best_loss(self, loss: float) -> bool:
        """Update best loss and return True if new best"""
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            return True
        return False
    
    def checkpoint_step(self):
        """Mark current step as last checkpointed"""
        self.last_checkpoint_step = self.global_step
    
    def filename_string(self) -> str:
        """Generate filename-safe progress string"""
        return f"step{self.global_step}-epoch{self.epoch}-batch{self.epoch_step}"
    
    def get_progress_dict(self) -> dict:
        """Get progress metrics as dictionary"""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "samples": self.epoch_sample,
            "samples_per_sec": self.samples_per_second,
            "best_loss": self.best_loss
        }