import time
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
from contextlib import contextmanager
from configs.training_config import TrainingConfig


@dataclass
class ProfileMetrics:
    """Container for profiling metrics"""
    batch_times: list[float]
    batch_sizes: list[int]
    memory_usage: list[float]
    losses: list[float]
    
    def clear(self):
        """Reset all metrics"""
        self.batch_times.clear()
        self.batch_sizes.clear()
        self.memory_usage.clear()
        self.losses.clear()
    
    def get_averages(self) -> Dict[str, float]:
        """Calculate average metrics over collected samples"""
        return {
            "avg_batch_time": np.mean(self.batch_times),
            "avg_throughput": np.mean(self.batch_sizes) / np.mean(self.batch_times),
            "avg_memory_gb": np.mean(self.memory_usage) / (1024 ** 3),
            "avg_loss": np.mean(self.losses)
        }


class TrainingProfiler:
    """Profiler for tracking training performance metrics"""
    
    def __init__(self, window_size: int = 50, config: Optional[TrainingConfig] = None):
        self.window_size = window_size
        self.metrics = ProfileMetrics([], [], [], [])
        self.current_range: Optional[str] = None
        self.range_start_time: Optional[float] = None
        self.config = config
        self.memory_callback: Optional[Callable[[], float]] = None
    
    def add_memory_callback(self, callback: Callable[[], float]):
        """Add callback for getting current memory usage"""
        self.memory_callback = callback
    
    @contextmanager
    def profile_range(self, range_name: str):
        """Context manager for profiling a specific code range"""
        try:
            self.current_range = range_name
            self.range_start_time = time.perf_counter()
            yield
        finally:
            self.current_range = None
            self.range_start_time = None
    
    @contextmanager
    def start_profiling(self):
        """Context manager for the entire profiling session"""
        try:
            self.metrics.clear()
            yield self
        finally:
            pass
    
    def record_step(self, batch_time: float, batch_size: int, memory_used: float, loss: float):
        """Record metrics for a training step"""
        self.metrics.batch_times.append(batch_time)
        self.metrics.batch_sizes.append(batch_size)
        
        # Use memory callback if available, otherwise use provided value
        if self.memory_callback is not None:
            memory_used = self.memory_callback()
        self.metrics.memory_usage.append(memory_used)
        
        self.metrics.losses.append(loss)
        
        # Keep only recent samples
        if len(self.metrics.batch_times) > self.window_size:
            self.metrics.batch_times.pop(0)
            self.metrics.batch_sizes.pop(0)
            self.metrics.memory_usage.pop(0)
            self.metrics.losses.pop(0)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current average metrics"""
        if not self.metrics.batch_times:  # Check if we have any metrics
            return {
                "avg_batch_time": 0.0,
                "avg_throughput": 0.0,
                "avg_memory_gb": 0.0 if not self.memory_callback else self.memory_callback() / (1024 ** 3),
                "avg_loss": 0.0
            }
        return self.metrics.get_averages()


class AutoTuner:
    """Automatic hyperparameter tuner based on training metrics"""
    
    def __init__(
        self,
        initial_params: Dict[str, Any],
        config: Optional[TrainingConfig] = None,
        min_samples: int = 50,
        throughput_threshold: float = 0.9,
        memory_threshold: float = 0.90
    ):
        self.current_params = initial_params.copy()
        self.best_params = initial_params.copy()
        self.best_throughput = 0.0
        self.min_samples = min_samples
        self.throughput_threshold = throughput_threshold
        self.memory_threshold = memory_threshold
        self.stable_steps = 0
        self.config = config
        
        # Initialize with config values if provided
        if config:
            self.current_params.update({
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.grad_accum_steps
            })
            self.best_params = self.current_params.copy()
        
    def update(self, profiler: TrainingProfiler) -> Dict[str, Any]:
        """Update hyperparameters based on profiling metrics"""
        metrics = profiler.get_current_metrics()
        
        # Wait for enough samples
        if len(profiler.metrics.batch_times) < self.min_samples:
            return self.current_params
        
        current_throughput = metrics["avg_throughput"]
        memory_usage = metrics["avg_memory_gb"]
        
        # Update best parameters if we found better throughput
        if current_throughput > self.best_throughput:
            self.best_throughput = current_throughput
            self.best_params = self.current_params.copy()
            self.stable_steps = 0
        else:
            self.stable_steps += 1
        
        # Adjust parameters based on metrics
        new_params = self.current_params.copy()
        
        # Adjust batch size based on memory usage and config limits
        if "batch_size" in new_params:
            max_batch_size = getattr(self.config, 'max_batch_size', 64)  # Default max batch size
            if memory_usage > self.memory_threshold * 80:  # 80GB GPU
                new_params["batch_size"] = max(1, new_params["batch_size"] - 1)
            elif memory_usage < self.memory_threshold * 65:  # More aggressive scaling for 80GB
                if new_params["batch_size"] < max_batch_size:
                    new_params["batch_size"] += 1
                    # Allow larger increases when far below threshold
                    if memory_usage < self.memory_threshold * 40:
                        new_params["batch_size"] = min(
                            new_params["batch_size"] + 1,
                            max_batch_size
                        )
        
        # Adjust gradient accumulation steps based on throughput and config
        if "gradient_accumulation_steps" in new_params:
            max_grad_steps = getattr(self.config, 'max_grad_accum_steps', 8)
            if current_throughput < self.best_throughput * self.throughput_threshold:
                new_params["gradient_accumulation_steps"] = max(
                    1, 
                    new_params["gradient_accumulation_steps"] - 1
                )
            elif self.stable_steps > 50:  # More aggressive adjustment for higher capacity GPU
                if new_params["gradient_accumulation_steps"] < max_grad_steps:
                    new_params["gradient_accumulation_steps"] += 1
                    self.stable_steps = 0
                    # Allow larger step increases when performance is stable
                    if self.stable_steps > 200:
                        new_params["gradient_accumulation_steps"] = min(
                            new_params["gradient_accumulation_steps"] + 2,
                            max_grad_steps
                        )
        
        # Update current parameters if changes were made
        if new_params != self.current_params:
            self.current_params = new_params
            self.stable_steps = 0
        
        return self.current_params