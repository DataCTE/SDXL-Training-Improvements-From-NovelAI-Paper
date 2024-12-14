import time
from typing import Dict, Optional, Any, Callable, Deque
from dataclasses import dataclass, field
import numpy as np
from collections import deque
from contextlib import contextmanager
from configs.training_config import TrainingConfig
import torch


@dataclass
class ProfileMetrics:
    """Optimized container for profiling metrics using fixed-size deques"""
    batch_times: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    batch_sizes: Deque[int] = field(default_factory=lambda: deque(maxlen=50))
    memory_usage: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    losses: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    gpu_utilization: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    throughput: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    
    def clear(self):
        """Reset all metrics efficiently"""
        for metric in (self.batch_times, self.batch_sizes, self.memory_usage, 
                      self.losses, self.gpu_utilization, self.throughput):
            metric.clear()
    
    def get_averages(self) -> Dict[str, float]:
        """Calculate average metrics with numpy vectorization"""
        if not self.batch_times:
            return {
                "avg_batch_time": 0.0,
                "avg_throughput": 0.0,
                "avg_memory_gb": 0.0,
                "avg_loss": 0.0,
                "gpu_util_percent": 0.0,
                "loss_std": 0.0,
                "throughput_std": 0.0
            }
            
        # Convert deques to numpy arrays for efficient computation
        times = np.array(self.batch_times)
        sizes = np.array(self.batch_sizes)
        losses = np.array(self.losses)
        memory = np.array(self.memory_usage)
        gpu_util = np.array(self.gpu_utilization)
        throughput = np.array(self.throughput)
        
        # Compute statistics efficiently
        return {
            "avg_batch_time": np.mean(times),
            "avg_throughput": np.sum(sizes) / np.sum(times),
            "avg_memory_gb": np.mean(memory) / (1024 ** 3),
            "avg_loss": np.mean(losses),
            "gpu_util_percent": np.mean(gpu_util),
            "loss_std": np.std(losses),
            "throughput_std": np.std(throughput),
            "memory_std_gb": np.std(memory) / (1024 ** 3)
        }


class TrainingProfiler:
    """Enhanced profiler for tracking training performance metrics"""
    
    def __init__(
        self, 
        window_size: int = 50,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.window_size = window_size
        self.metrics = ProfileMetrics()
        self.current_range: Optional[str] = None
        self.range_start_time: Optional[float] = None
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_callback: Optional[Callable[[], float]] = None
        self.gpu_util_callback: Optional[Callable[[], float]] = None
        
        # Initialize CUDA events for precise GPU timing
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        
        # Add memory tracking
        self.memory_tracker = torch.cuda.memory.memory_stats()
    
    def add_memory_callback(self, callback: Callable[[], float]):
        """Add callback for getting current memory usage"""
        self.memory_callback = callback
    
    def add_gpu_util_callback(self, callback: Callable[[], float]):
        """Add callback for getting GPU utilization"""
        self.gpu_util_callback = callback
    
    @contextmanager
    def profile_range(self, range_name: str):
        """Context manager for profiling a specific code range with CUDA events"""
        try:
            self.current_range = range_name
            if torch.cuda.is_available():
                self.start_event.record()
            else:
                self.range_start_time = time.perf_counter()
            yield
        finally:
            if torch.cuda.is_available():
                self.end_event.record()
                self.end_event.synchronize()
                elapsed = self.start_event.elapsed_time(self.end_event) / 1000.0  # Convert to seconds
            else:
                elapsed = time.perf_counter() - self.range_start_time
                
            self.current_range = None
            self.range_start_time = None
    
    @contextmanager
    def start_profiling(self):
        """Context manager for the entire profiling session with automatic cleanup"""
        try:
            self.metrics.clear()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
            yield self
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def record_step(
        self, 
        batch_time: float, 
        batch_size: int, 
        memory_used: float, 
        loss: float,
        throughput: Optional[float] = None
    ):
        """Record metrics for a training step with enhanced memory tracking"""
        self.metrics.batch_times.append(batch_time)
        self.metrics.batch_sizes.append(batch_size)
        
        # Use memory callback if available
        if self.memory_callback is not None:
            memory_used = self.memory_callback()
        self.metrics.memory_usage.append(memory_used)
        
        # Track GPU utilization if callback available
        if self.gpu_util_callback is not None:
            self.metrics.gpu_utilization.append(self.gpu_util_callback())
        
        self.metrics.losses.append(loss)
        
        # Calculate and record throughput
        if throughput is None and batch_time > 0:
            throughput = batch_size / batch_time
        if throughput is not None:
            self.metrics.throughput.append(throughput)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current average metrics with memory peak tracking"""
        base_metrics = self.metrics.get_averages()
        
        # Add peak memory usage if available
        if torch.cuda.is_available():
            base_metrics["peak_memory_gb"] = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            )
            base_metrics["peak_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
            )
        
        return base_metrics


class AutoTuner:
    """Enhanced automatic hyperparameter tuner with improved memory management"""
    
    def __init__(
        self,
        initial_params: Dict[str, Any],
        config: Optional[TrainingConfig] = None,
        min_samples: int = 50,
        throughput_threshold: float = 0.9,
        memory_threshold: float = 0.90,
        device: Optional[torch.device] = None
    ):
        self.current_params = initial_params.copy()
        self.best_params = initial_params.copy()
        self.best_throughput = 0.0
        self.min_samples = min_samples
        self.throughput_threshold = throughput_threshold
        self.memory_threshold = memory_threshold
        self.stable_steps = 0
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize with config values if provided
        if config:
            self.current_params.update({
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.grad_accum_steps
            })
            self.best_params = self.current_params.copy()
        
        # Track parameter history
        self.param_history = deque(maxlen=100)
        
    def update(self, profiler: TrainingProfiler) -> Dict[str, Any]:
        """Update hyperparameters based on profiling metrics with improved stability"""
        metrics = profiler.get_current_metrics()
        
        # Wait for enough samples
        if len(profiler.metrics.batch_times) < self.min_samples:
            return self.current_params
        
        current_throughput = metrics["avg_throughput"]
        memory_usage = metrics["avg_memory_gb"]
        memory_std = metrics.get("memory_std_gb", 0.0)
        throughput_std = metrics.get("throughput_std", 0.0)
        
        # Update best parameters if we found better throughput with stability check
        if (current_throughput > self.best_throughput and 
            throughput_std / current_throughput < 0.1):  # Ensure stable performance
            self.best_throughput = current_throughput
            self.best_params = self.current_params.copy()
            self.stable_steps = 0
        else:
            self.stable_steps += 1
        
        # Adjust parameters based on metrics
        new_params = self.current_params.copy()
        
        # Get memory limits based on device
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
        memory_threshold = self.memory_threshold * total_memory
        
        # Adjust batch size based on memory usage and stability
        if "batch_size" in new_params:
            max_batch_size = getattr(self.config, 'max_batch_size', 64)
            
            if memory_usage + 2 * memory_std > memory_threshold:
                new_params["batch_size"] = max(1, new_params["batch_size"] - 1)
            elif (memory_usage + 3 * memory_std < memory_threshold * 0.8 and 
                  throughput_std / current_throughput < 0.15):
                if new_params["batch_size"] < max_batch_size:
                    new_params["batch_size"] += 1
        
        # Adjust gradient accumulation steps based on throughput stability
        if "gradient_accumulation_steps" in new_params:
            max_grad_steps = getattr(self.config, 'max_grad_accum_steps', 8)
            
            if (current_throughput < self.best_throughput * self.throughput_threshold or 
                throughput_std / current_throughput > 0.2):
                new_params["gradient_accumulation_steps"] = max(
                    1, 
                    new_params["gradient_accumulation_steps"] - 1
                )
            elif (self.stable_steps > 50 and 
                  throughput_std / current_throughput < 0.1):
                if new_params["gradient_accumulation_steps"] < max_grad_steps:
                    new_params["gradient_accumulation_steps"] += 1
                    self.stable_steps = 0
        
        # Track parameter changes
        if new_params != self.current_params:
            self.param_history.append((self.current_params.copy(), metrics))
            self.current_params = new_params
            self.stable_steps = 0
        
        return self.current_params