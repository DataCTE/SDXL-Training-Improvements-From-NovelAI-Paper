"""EDM Euler scheduler implementation with NAI improvements."""

import torch
import numpy as np
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
import math


@dataclass
class EDMEulerSchedulerConfig:
    """Configuration for the EDMEulerScheduler."""
    
    num_train_timesteps: int = 1000
    sigma_min: float = 0.002  # NAI minimum sigma
    sigma_max: float = 20000.0  # NAI practical infinity
    sigma_data: float = 1.0  # Standard deviation of data distribution
    s_noise: float = 1.0  # Amount of additional noise to add during sampling
    s_churn: float = 0.0  # Parameters for stochastic sampling
    s_tmin: float = 0.0  # Minimum number of timesteps
    s_tmax: float = float('inf')  # Maximum number of timesteps
    prediction_type: str = "v_prediction"  # NAI uses v-prediction


class EDMEulerScheduler(SchedulerMixin):
    """
    EDM (Elucidating Denoising Models) scheduler with NAI improvements.
    Implements Zero Terminal SNR and proper sigma scaling.
    """
    
    config_name = "edm_euler_scheduler"
    _compatibles = ["stable_diffusion_xl"]
    order = 1

    def __init__(self, **kwargs):
        self.config = EDMEulerSchedulerConfig(**kwargs)
        
        # Create timesteps according to EDM formulation
        self.timesteps = self._generate_timesteps()
        
        # Generate sigmas for training
        self.sigmas = self._generate_sigmas()
        
        # Initialize state
        self.num_inference_steps = None
        self.cur_model_output = None
        self.counter = 0

    def _generate_timesteps(self) -> torch.Tensor:
        """Generate timesteps according to EDM paper."""
        steps = torch.arange(0, self.config.num_train_timesteps, dtype=torch.float32)
        return steps

    def _generate_sigmas(self) -> torch.Tensor:
        """Generate noise levels (sigmas) with NAI's practical infinity."""
        # Generate log-linear spacing from sigma_min to sigma_max
        rho = 7.0  # EDM's recommended value
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max
        
        sigmas = []
        for i in range(self.config.num_train_timesteps):
            t = i / (self.config.num_train_timesteps - 1)
            log_sigma = math.log(sigma_max) + t * (math.log(sigma_min) - math.log(sigma_max))
            sigmas.append(math.exp(log_sigma))
            
        return torch.tensor(sigmas, dtype=torch.float32)

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        Scales the denoising model input by the standard deviation of the noise level.
        Args:
            sample: Input sample
            timestep: Current timestep
        Returns:
            Scaled input sample
        """
        if timestep is None:
            return sample
            
        sigma = self.sigmas[timestep]
        sample = sample / ((sigma ** 2 + self.config.sigma_data ** 2) ** 0.5)
        return sample

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the number of timesteps for inference.
        Args:
            num_inference_steps: Number of inference steps
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = self._generate_timesteps()
        self.sigmas = self._generate_sigmas()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the EDM process.
        Args:
            model_output: Direct output from trained model (v-prediction)
            timestep: Current discrete timestep
            sample: Current instance of sample being created
            return_dict: Whether to return a SchedulerOutput instead of tuple
        Returns:
            prev_sample
        """
        sigma = self.sigmas[timestep]
        sigma_next = self.sigmas[timestep + 1] if timestep < len(self.sigmas) - 1 else 0.0
        
        # Convert to EDM's internal timestep representation
        t = -torch.log(sigma)
        t_next = -torch.log(torch.tensor(sigma_next))
        
        # Calculate stochastic sampling parameters
        gamma = min(self.config.s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1) if self.config.s_tmin <= t <= self.config.s_tmax else 0.0
        
        # Apply stochastic sampling if enabled
        if gamma > 0:
            noise = torch.randn_like(sample)
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                sample = sample + noise * (sigma_hat ** 2 - sigma ** 2) ** 0.5
            sigma = sigma_hat
        
        # Compute denoised sample using v-prediction
        # v = (x - denoised) * sigma / (sigma^2 + 1)
        # Therefore: denoised = x - v * sigma * (sigma^2 + 1)
        denoised = sample - model_output * sigma * (sigma ** 2 + 1)
        
        # Euler step
        d = (sample - denoised) / sigma
        dt = t_next - t
        sample_next = sample + dt * d
        
        # Store for future steps
        self.cur_model_output = model_output
        
        if not return_dict:
            return (sample_next,)
            
        return SchedulerOutput(prev_sample=sample_next)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples according to EDM formulation.
        Args:
            original_samples: Clean samples
            noise: Random noise
            timesteps: Timesteps at which to add noise
        Returns:
            Noisy samples
        """
        sigmas = self.sigmas[timesteps].to(device=original_samples.device)
        noisy_samples = original_samples + noise * sigmas.view(-1, 1, 1, 1)
        return noisy_samples