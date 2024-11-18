"""Progress tracking utilities with integrated logging."""

import wandb
import logging
import json
from typing import Optional, Any, Dict, Union, List, Callable
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
                auto_refresh=False
            )
            self._active_tasks = []
            self._is_started = False
            self._task_stack = []
            self._callbacks = {}
    
    def add_task(self, description: str, total: Optional[float] = None) -> TaskID:
        """Add a new task to the progress manager."""
        task_id = self.progress.add_task(description, total=total)
        self._active_tasks.append(task_id)
        return task_id
    
    def update(self, task_id: TaskID, advance: float = 1):
        """Update task progress."""
        self.progress.update(task_id, advance=advance)
        self._trigger_callback(task_id, "update", advance=advance)
    
    def remove_task(self, task_id: TaskID):
        """Remove a task from the progress manager."""
        if task_id in self._active_tasks:
            self.progress.remove_task(task_id)
            self._active_tasks.remove(task_id)
            self._trigger_callback(task_id, "remove")
    
    def add_callback(self, task_id: TaskID, event: str, callback: Callable):
        """Add a callback for task events."""
        if task_id not in self._callbacks:
            self._callbacks[task_id] = {}
        self._callbacks[task_id][event] = callback
    
    def _trigger_callback(self, task_id: TaskID, event: str, **kwargs):
        """Trigger callback if it exists."""
        if task_id in self._callbacks and event in self._callbacks[task_id]:
            self._callbacks[task_id][event](**kwargs)

class ProgressTracker:
    """Enhanced progress tracker with metrics logging and callbacks."""
    
    def __init__(
        self,
        description: str,
        total: Optional[float] = None,
        wandb_run: Optional[Any] = None,
        log_interval: float = 0.1,
        log_dir: Optional[Union[str, Path]] = None,
        total_steps: Optional[int] = None,
        log_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        eval_steps: Optional[int] = None
    ):
        self.description = description
        self.wandb_run = wandb_run
        self.log_interval = log_interval
        self.log_dir = Path(log_dir) if log_dir else None
        
        # Training specific steps
        self.total_steps = total_steps
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.current_step = 0
        
        # Setup logging directory and files
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = self.log_dir / f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            self.state_file = self.log_dir / "latest_state.json"
        
        # Get progress manager instance
        self.manager = ProgressManager()
        
        # Add task with proper total
        self.task_id = self.manager.add_task(
            description,
            total=total or total_steps or float('inf')
        )
        
        # Initialize metrics history
        self.metrics_history = []
        
        # Log initial state
        self._log_initial_state()
    
    def _log_initial_state(self):
        """Log initial training state."""
        initial_state = {
            "description": self.description,
            "total_steps": self.total_steps,
            "log_steps": self.log_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "start_time": datetime.now().isoformat()
        }
        
        logger.info(f"Started progress tracking: {self.description}")
        if self.total_steps:
            logger.info(f"Total steps: {self.total_steps}")
        
        if self.log_dir:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(initial_state, f, indent=2)
    
    def _log_to_file(self, metrics: Dict[str, Any]):
        """Log metrics to JSONL file."""
        if not self.log_dir:
            return
            
        entry = {
            "step": self.current_step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    
    def should_log(self, step: int) -> bool:
        """Check if we should log metrics at current step."""
        return self.log_steps and step % self.log_steps == 0
    
    def should_save(self, step: int) -> bool:
        """Check if we should save checkpoint at current step."""
        return self.save_steps and step % self.save_steps == 0
    
    def should_evaluate(self, step: int) -> bool:
        """Check if we should run evaluation at current step."""
        return self.eval_steps and step % self.eval_steps == 0
    
    def update(
        self,
        advance: float = 1,
        metrics: Optional[Dict[str, Any]] = None,
        force_log: bool = False
    ):
        """Update progress and log metrics."""
        self.current_step += advance
        self.manager.update(self.task_id, advance=advance)
        
        if metrics and (force_log or self.should_log(self.current_step)):
            # Add step to metrics
            metrics["step"] = self.current_step
            
            # Log to console
            metrics_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in metrics.items())
            logger.info(f"{self.description} - Step {self.current_step}: {metrics_str}")
            
            # Log to file
            self._log_to_file(metrics)
            
            # Log to wandb
            if self.wandb_run is not None:
                self.wandb_run.log(metrics, step=self.current_step)
            
            # Store in history
            self.metrics_history.append(metrics)
    
    def finish(self, final_metrics: Optional[Dict[str, Any]] = None):
        """Complete the progress tracking with final metrics."""
        if final_metrics:
            self.update(metrics=final_metrics, force_log=True)
        
        # Save final state
        if self.log_dir:
            final_state = {
                "total_steps_completed": self.current_step,
                "end_time": datetime.now().isoformat(),
                "final_metrics": final_metrics
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(final_state, f, indent=2)
        
        logger.info(f"{self.description} completed after {self.current_step} steps")
        self.manager.remove_task(self.task_id)
        
def create_progress_tracker(
    description: str,
    total_steps: Optional[int] = None,
    log_steps: Optional[int] = None,
    save_steps: Optional[int] = None,
    eval_steps: Optional[int] = None,
    wandb_run: Optional[Any] = None,
    log_dir: Optional[Union[str, Path]] = None,
    log_interval: float = 0.1
) -> ProgressTracker:
    """Create a progress tracker with training-specific configuration."""
    return ProgressTracker(
        description=description,
        total_steps=total_steps,
        log_steps=log_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        wandb_run=wandb_run,
        log_dir=log_dir,
        log_interval=log_interval
    )