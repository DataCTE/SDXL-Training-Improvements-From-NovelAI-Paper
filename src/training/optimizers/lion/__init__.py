"""Lion optimizer implementation with NAI improvements."""

import torch
from torch.optim.optimizer import Optimizer
from typing import Tuple, Optional, Callable
import math


class Lion(Optimizer):
    """
    Lion optimizer with NAI's recommended settings.
    Implements the optimizer from "Symbolic Discovery of Optimization Algorithms" 
    (https://arxiv.org/abs/2302.06675) with improvements from NAI research.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.95, 0.98),  # NAI recommended values
        weight_decay: float = 0.0,
        optimize_memory: bool = True,
    ):
        """Initialize Lion optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (NAI recommends 1e-4 to 3e-4)
            betas: Coefficients for computing moving averages
            weight_decay: Weight decay factor
            optimize_memory: Whether to use memory optimizations
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            optimize_memory=optimize_memory,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step with proper beta2 usage."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get gradient and state
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)  # Add second moment estimate
                    state["update_direction"] = torch.zeros_like(p)
                    state["steps"] = 0

                # Get parameters
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]  # Second moment estimate
                update_direction = state["update_direction"]
                beta1, beta2 = group["betas"]
                
                # Weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad.square(), alpha=1 - beta2)  # Update second moment

                # Bias correction
                bias_correction1 = 1 - beta1 ** (state["steps"] + 1)
                bias_correction2 = 1 - beta2 ** (state["steps"] + 1)
                
                # Compute update with bias correction and second moment
                update = exp_avg / bias_correction1
                if state["steps"] % 2 == 0:
                    update = update.sign()
                else:
                    # Use second moment for odd steps
                    update = (update / (exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + 1e-8)).sign()
                
                # Memory optimization
                if group["optimize_memory"]:
                    update_direction.copy_(update)
                    p.add_(update_direction, alpha=-group["lr"])
                else:
                    p.add_(update, alpha=-group["lr"])

                state["steps"] += 1

        return loss
