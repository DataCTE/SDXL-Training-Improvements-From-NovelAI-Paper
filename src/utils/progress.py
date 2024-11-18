# src/utils/progress.py
import wandb
from typing import Optional, Any, Dict
from pip._vendor.rich.progress import (
    Progress, 
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)

class ProgressTracker:
    def __init__(
        self, 
        description: str,
        total: Optional[float] = None,
        wandb_run: Optional[wandb.Run] = None,
        log_interval: float = 0.1
    ):
        self.description = description
        self.wandb_run = wandb_run
        self.log_interval = log_interval
        
        # Create Rich progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            refresh_per_second=10,
            transient=False
        )
        
        # Start progress tracking
        self.progress.start()
        self.task_id = self.progress.add_task(
            description,
            total=total
        )
        
    def update(self, advance: float = 1, metrics: Optional[Dict[str, Any]] = None):
        """Update both progress bar and wandb metrics"""
        self.progress.update(self.task_id, advance=advance)
        
        if self.wandb_run is not None and metrics:
            self.wandb_run.log(metrics)
            
    def finish(self, final_metrics: Optional[Dict[str, Any]] = None):
        """Complete the progress tracking"""
        if final_metrics and self.wandb_run:
            self.wandb_run.log(final_metrics)
        self.progress.stop()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()