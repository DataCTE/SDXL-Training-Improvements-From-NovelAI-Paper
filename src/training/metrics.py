import logging
from dataclasses import dataclass
from collections import defaultdict
from multiprocessing import Manager
from typing import Dict, Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class AverageMeter:
    """Fully pickleable average meter."""
    name: str
    fmt: str = ":f"
    window_size: Optional[int] = None
    
    def __post_init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self._lock = Lock()
        
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]
        return state
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = Lock()
        
    def reset(self):
        with self._lock:
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
            
    def update(self, val, n=1):
        with self._lock:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

class MetricsManager:
    """Fully process-safe metrics manager."""
    
    def __init__(self):
        self._manager = Manager()
        self._metrics = self._manager.dict()
        
    def get_metric(self, name: str) -> AverageMeter:
        """Get or create a metric by name."""
        if name not in self._metrics:
            self._metrics[name] = AverageMeter(name)
        return self._metrics[name]
        
    def update_metric(self, name: str, value: float, n: int = 1):
        """Update a metric with a new value."""
        metric = self.get_metric(name)
        metric.update(value, n)
        
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all metrics as a dictionary of averages."""
        return {name: metric.avg for name, metric in self._metrics.items()}
