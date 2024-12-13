import torch
import torch.cuda
import torch.cuda.memory
import gc

class MemoryManager:
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        self.oom_count = 0
        
    def update(self):
        self.current_memory = torch.cuda.memory_allocated()
        self.peak_memory = max(self.peak_memory, self.current_memory)
        
    def clear_cache(self):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
    def handle_oom(self):
        self.oom_count += 1
        self.clear_cache()
        return self.oom_count > 3  # Return True if OOM is persistent
        
    def log_memory_stats(self):
        """Log current memory statistics"""
        print(f"Memory Stats:")
        print(f"  Current: {self.current_memory/1e9:.1f}GB")
        print(f"  Peak: {self.peak_memory/1e9:.1f}GB")
        print(f"  OOM Count: {self.oom_count}")
