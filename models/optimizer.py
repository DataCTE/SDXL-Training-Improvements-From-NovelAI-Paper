import torch
from adamw_bf16 import AdamW_BF16 as BaseBF16Optimizer
from typing import Optional, Callable
from utils.error_handling import error_handler
from torch.distributed import dist

class SDXLAdamWBF16(BaseBF16Optimizer):
    """
    Extended AdamW_BF16 optimizer specifically for SDXL training.
    Inherits from the base AdamW_BF16 optimizer and adds SDXL-specific optimizations.
    
    Features:
    - Keeps weights in bfloat16 with correction terms
    - Implements stochastic rounding for better precision
    - Adds noise scaling for SDXL training
    - Includes gradient clipping
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
        noise_scale: float = 1.0,
        lr_function: Optional[Callable[[int], float]] = None,
        **kwargs
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            lr_function=lr_function,
            **kwargs
        )
        self.max_grad_norm = max_grad_norm
        self.noise_scale = noise_scale

    @error_handler
    @torch.no_grad()
    def step(self, closure=None, zero_grad: bool = False):
        """
        Performs a single optimization step with gradient clipping and noise scaling.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
            zero_grad (bool): Whether to zero gradients after step. More efficient than separate calls.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Apply gradient clipping if specified
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.param_groups for p in group['params']],
                self.max_grad_norm
            )

        # Scale gradients by noise level before optimization
        if self.noise_scale != 1.0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data *= self.noise_scale

        # Add check for gradient synchronization
        if dist.is_initialized():
            # Only average gradients on final accumulation step
            if (self.global_step + 1) % self.grad_accum_steps == 0:
                # Ensure all processes are ready
                dist.barrier()
                for param in self.param_groups[0]['params']:
                    if param.grad is not None:
                        # In-place operation to save memory
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data.div_(dist.get_world_size())

        # Call parent class step (handles bfloat16 conversion and correction terms)
        super().step(closure=None)
        
        # Optionally zero gradients (more efficient than separate call)
        if zero_grad:
            self.zero_grad(set_to_none=True)

        return loss

    @error_handler
    def zero_grad(self, set_to_none: bool = True):
        """
        Resets the gradients of all optimized parameters.
        
        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                              This will in general have lower memory footprint.
        """
        super().zero_grad(set_to_none=set_to_none) 