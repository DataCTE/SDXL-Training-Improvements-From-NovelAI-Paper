"""Progress tracking utilities with integrated logging."""

import wandb
import logging
from typing import Optional, Any, Dict, Union, List
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

class ProgressManager:
    """Manages multiple progress bars."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__()
        return cls._instance
    
    def __init__(self):
        """Initialize the progress manager if not already initialized."""
        if not hasattr(self, 'progress'):
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                refresh_per_second=10,
                expand=True
            )
            self._active_tasks = []
            self._is_started = False
    
    @property
    def active_tasks(self) -> List[TaskID]:
        """Get list of active task IDs."""
        return self._active_tasks
    
    @property
    def is_started(self) -> bool:
        """Get progress display start status."""
        return self._is_started
    
    @is_started.setter
    def is_started(self, value: bool):
        """Set progress display start status."""
        self._is_started = value
    
    def start(self):
        """Start the progress display if not already started."""
        if not self.is_started:
            self.progress.start()
            self.is_started = True
    
    def stop(self):
        """Stop the progress display if started and no active tasks."""
        if self.is_started and not self.active_tasks:
            self.progress.stop()
            self.is_started = False
    
    def add_task(self, *args, **kwargs) -> TaskID:
        """Add a new task and track it."""
        task_id = self.progress.add_task(*args, **kwargs)
        self._active_tasks.append(task_id)
        return task_id
    
    def remove_task(self, task_id: TaskID):
        """Remove a task from tracking."""
        if task_id in self._active_tasks:
            self._active_tasks.remove(task_id)
            if not self._active_tasks:
                self.stop()

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
        
        # Get progress manager instance
        self.manager = ProgressManager()
        self.manager.start()
        
        # Add task
        self.task_id = self.manager.add_task(
            description,
            total=total if total is not None else float('inf')
        )
        
        # Log initial state
        logger.info(f"Started progress tracking: {description}")
        if total:
            logger.info(f"Total steps: {total}")
    
    def update(self, advance: float = 1, metrics: Optional[Dict[str, Any]] = None):
        """Update progress bar, logs, and wandb metrics"""
        self.manager.progress.update(task_id=self.task_id, advance=advance)
        
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
    
    def finish(self, final_metrics: Optional[Dict[str, Any]] = None):
        """Complete the progress tracking"""
        if final_metrics:
            metrics_str = ", ".join(f"{k}: {v}" for k, v in final_metrics.items())
            logger.info(f"{self.description} completed - Final metrics: {metrics_str}")
            
            if self.log_dir:
                self._log_to_file(final_metrics)
            
            if self.wandb_run is not None:
                self.wandb_run.log(final_metrics)
        else:
            logger.info(f"{self.description} completed")
        
        self.manager.remove_task(self.task_id)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
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