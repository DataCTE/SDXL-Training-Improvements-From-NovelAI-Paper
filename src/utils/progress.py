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
                expand=True,
                auto_refresh=False  # Important: disable auto refresh
            )
            self._active_tasks = []
            self._is_started = False
            self._task_stack = []  # New: stack to track nested tasks
    
    @property
    def current_task(self) -> Optional[TaskID]:
        """Get the current active task."""
        return self._task_stack[-1] if self._task_stack else None
    
    def start(self):
        """Start the progress display if not already started."""
        if not self._is_started:
            self.progress.start()
            self._is_started = True
    
    def stop(self):
        """Stop the progress display if started and no active tasks."""
        if self._is_started and not self._active_tasks:
            self.progress.stop()
            self._is_started = False
    
    def add_task(self, description: str, total: Optional[float] = None, **kwargs) -> TaskID:
        """Add a new task and track it."""
        # Create new task
        task_id = self.progress.add_task(
            description=description,
            total=total if total is not None else float('inf'),
            **kwargs
        )
        
        # Track the task
        self._active_tasks.append(task_id)
        self._task_stack.append(task_id)
        
        # Start progress if needed
        if not self._is_started:
            self.start()
        
        return task_id
    
    def remove_task(self, task_id: TaskID):
        """Remove a task from tracking."""
        if task_id in self._active_tasks:
            self._active_tasks.remove(task_id)
            if task_id in self._task_stack:
                self._task_stack.remove(task_id)
            
            # Stop if no more tasks
            if not self._active_tasks:
                self.stop()
    
    def update(self, task_id: TaskID, advance: float = 1, **kwargs):
        """Update a task's progress."""
        self.progress.update(task_id, advance=advance, **kwargs)
        self.progress.refresh()  # Manual refresh

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
        
        # Add task
        self.task_id = self.manager.add_task(description, total=total)
        
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