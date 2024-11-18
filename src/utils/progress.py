"""Progress tracking utilities with integrated logging."""

import wandb
import logging
from typing import Optional, Any, Dict, Union
from pathlib import Path
from datetime import datetime
from pip._vendor.rich.progress import (
    Progress, 
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TaskID
)

logger = logging.getLogger(__name__)

class ProgressTracker:
    def __init__(
        self, 
        description: str,
        total: Optional[float] = None,
        wandb_run: Optional[Any] = None,
        log_interval: float = 0.1,
        log_dir: Optional[Union[str, Path]] = None
    ):
        self.description = description
        self.wandb_run = wandb_run
        self.log_interval = log_interval
        self.log_dir = Path(log_dir) if log_dir else None
        
        # Setup logging
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = self.log_dir / f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
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
        self.task_id: TaskID = self.progress.add_task(
            description,
            total=total if total is not None else float('inf')
        )
        
        # Log initial state
        logger.info(f"Started progress tracking: {description}")
        if total:
            logger.info(f"Total steps: {total}")
        
    def update(self, advance: float = 1, metrics: Optional[Dict[str, Any]] = None):
        """Update progress bar, logs, and wandb metrics"""
        self.progress.update(task_id=self.task_id, advance=advance)
        
        if metrics:
            # Log to console
            metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())
            logger.debug(f"{self.description} - Progress update: {metrics_str}")
            
            # Log to file if enabled
            if self.log_dir:
                self._log_to_file(metrics)
            
            # Log to wandb if enabled
            if self.wandb_run is not None:
                self.wandb_run.log(metrics)
                
    def _log_to_file(self, metrics: Dict[str, Any]):
        """Log metrics to JSON file"""
        if hasattr(self, 'metrics_file'):
            from json import dumps
            task = self.progress.tasks[int(self.task_id)]  # Convert TaskID to int for indexing
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "description": self.description,
                "progress": task.completed,
                "total": task.total,
                "metrics": metrics
            }
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(dumps(log_entry) + "\n")
            
    def finish(self, final_metrics: Optional[Dict[str, Any]] = None):
        """Complete the progress tracking"""
        if final_metrics:
            # Log final metrics
            metrics_str = ", ".join(f"{k}: {v}" for k, v in final_metrics.items())
            logger.info(f"{self.description} completed - Final metrics: {metrics_str}")
            
            # Log to file if enabled
            if self.log_dir:
                self._log_to_file(final_metrics)
            
            # Log to wandb if enabled
            if self.wandb_run is not None:
                self.wandb_run.log(final_metrics)
        else:
            logger.info(f"{self.description} completed")
            
        self.progress.stop()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Log any errors that occurred
            logger.error(f"Error in {self.description}: {str(exc_val)}")
        self.finish()

def create_progress_tracker(
    description: str,
    total: Optional[float] = None,
    wandb_run: Optional[Any] = None,
    log_dir: Optional[Union[str, Path]] = None,
    log_interval: float = 0.1
) -> ProgressTracker:
    """Factory function to create a progress tracker with proper logging setup."""
    return ProgressTracker(
        description=description,
        total=total,
        wandb_run=wandb_run,
        log_dir=log_dir,
        log_interval=log_interval
    )