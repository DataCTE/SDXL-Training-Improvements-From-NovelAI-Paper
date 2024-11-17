import math
import numpy as np
import scipy.stats
import torch
import comfy.samplers
import comfy.sample
import latent_preview
from torch.distributions import Laplace


def karras_scheduler(model, steps, sigma_min=0.0292, sigma_max=14.6146):
    """
    Karras scheduler from k-diffusion.

    Args:
        model: The diffusion model.
        steps (int): Number of steps in the scheduler.
        sigma_min (float): Minimum sigma value.
        sigma_max (float): Maximum sigma value.

    Returns:
        torch.Tensor: Scheduled sigma values.
    """
    rho = 7.0  # 7.0 is the value used in the paper
    ramp = np.linspace(0, 1, steps)
    sigmas = np.sqrt((sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho)
    sigmas = np.flip(sigmas).copy()
    sigmas = torch.from_numpy(sigmas).float()
    return torch.cat([sigmas, torch.tensor([0.0], device=sigmas.device)])


def exponential_scheduler(model, steps, sigma_min=0.0292, sigma_max=14.6146):
    """
    Exponential scheduler.

    Args:
        model: The diffusion model.
        steps (int): Number of steps in the scheduler.
        sigma_min (float): Minimum sigma value.
        sigma_max (float): Maximum sigma value.

    Returns:
        torch.Tensor: Scheduled sigma values.
    """
    ramp = np.linspace(0, 1, steps)
    sigmas = np.exp(np.log(sigma_max) + ramp * (np.log(sigma_min) - np.log(sigma_max)))
    sigmas = np.flip(sigmas).copy()
    sigmas = torch.from_numpy(sigmas).float()
    return torch.cat([sigmas, torch.tensor([0.0], device=sigmas.device)])


def sgm_uniform_scheduler(model, steps):
    """
    SGM uniform scheduler.

    Args:
        model: The diffusion model.
        steps (int): Number of steps in the scheduler.

    Returns:
        torch.Tensor: Scheduled sigma values.
    """
    return comfy.sample.get_sigmas_uniform(steps, model)


def simple_scheduler(model, steps):
    """
    Simple linear scheduler.

    Args:
        model: The diffusion model.
        steps (int): Number of steps in the scheduler.

    Returns:
        torch.Tensor: Scheduled sigma values.
    """
    return comfy.sample.get_sigmas_linear(steps, model)


def ddim_uniform_scheduler(model, steps):
    """
    DDIM uniform scheduler.

    Args:
        model: The diffusion model.
        steps (int): Number of steps in the scheduler.

    Returns:
        torch.Tensor: Scheduled sigma values.
    """
    return comfy.sample.get_sigmas_uniform(steps, model)


def normal_scheduler(model, steps, sigma_min=0.0292, sigma_max=14.6146):
    """
    Normal distribution scheduler.

    Args:
        model: The diffusion model.
        steps (int): Number of steps in the scheduler.
        sigma_min (float): Minimum sigma value.
        sigma_max (float): Maximum sigma value.

    Returns:
        torch.Tensor: Scheduled sigma values.
    """
    ramp = np.linspace(0, 1, steps)
    sigmas = scipy.stats.norm.ppf(1 - ramp) * (sigma_max - sigma_min) / 2 + sigma_min
    sigmas = np.flip(sigmas).copy()
    sigmas = torch.from_numpy(sigmas).float()
    return torch.cat([sigmas, torch.tensor([0.0], device=sigmas.device)])


def perlin_noise(grid_shape, out_shape, batch_size=1, generator=None):
    """
    Generate Perlin noise with given shape.

    Args:
        grid_shape (tuple): Grid height and width.
        out_shape (tuple): Output height and width.
        batch_size (int): Number of noise samples to generate.
        generator (torch.Generator, optional): Random number generator.

    Returns:
        torch.Tensor: Perlin noise tensor.
    """
    gh, gw = grid_shape
    oh, ow = out_shape
    bh, bw = oh // gh, ow // gw

    if oh != bh * gh:
        raise ValueError(f"Output height {oh} must be divisible by grid height {gh}")
    if ow != bw * gw:
        raise ValueError(f"Output width {ow} must be divisible by grid width {gw}")

    # Generate random angles
    angle = torch.empty([batch_size] + [s + 1 for s in grid_shape], device=generator.device if generator else 'cpu')
    angle.uniform_(0.0, 2.0 * math.pi, generator=generator)

    # Random vectors on grid points
    vectors = torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
    vectors = unfold_grid(vectors)

    # Positions inside grid cells [0, 1)
    positions = get_positions((bh, bw)).to(vectors.device)

    return perlin_noise_tensor(vectors, positions).squeeze(0)


def get_positions(block_shape):
    """
    Generate position tensor.

    Args:
        block_shape (tuple): Block height and width.

    Returns:
        torch.Tensor: Position tensor.
    """
    bh, bw = block_shape
    positions = torch.stack(
        torch.meshgrid(
            [torch.arange(b) + 0.5 for b in (bw, bh)],
            indexing="ij",
        ),
        -1,
    ).float().view(1, bh, bw, 1, 1, 2)
    return positions


def unfold_grid(vectors):
    """
    Unfold vector grid to batched vectors.

    Args:
        vectors (torch.Tensor): Vectors tensor.

    Returns:
        torch.Tensor: Unfolded vectors tensor.
    """
    batch_size, _, gpy, gpx = vectors.shape
    return (
        torch.nn.functional.unfold(vectors, (2, 2))
        .view(batch_size, 2, 4, -1)
        .permute(0, 2, 3, 1)
        .view(batch_size, 4, gpy - 1, gpx - 1, 2)
    )


def perlin_noise_tensor(vectors, positions):
    """
    Generate Perlin noise from batched vectors and positions.

    Args:
        vectors (torch.Tensor): Vectors tensor.
        positions (torch.Tensor): Positions tensor.

    Returns:
        torch.Tensor: Perlin noise tensor.
    """
    batch_size = vectors.shape[0]
    gh, gw = vectors.shape[2:4]
    bh, bw = positions.shape[1:3]

    vectors = vectors.view(batch_size, 4, 1, gh * gw, 2)
    positions = positions.view(positions.shape[0], bh * bw, -1, 2)

    step_x = smooth_step(positions[..., 0])
    step_y = smooth_step(positions[..., 1])

    # Compute dot products
    dot00 = (vectors[:, 0] * positions).sum(dim=-1)
    dot10 = (vectors[:, 1] * (positions - torch.tensor([1.0, 0.0], device=positions.device))) .sum(dim=-1)
    dot01 = (vectors[:, 2] * (positions - torch.tensor([0.0, 1.0], device=positions.device))) .sum(dim=-1)
    dot11 = (vectors[:, 3] * (positions - torch.tensor([1.0, 1.0], device=positions.device))) .sum(dim=-1)

    # Interpolate
    row0 = torch.lerp(dot00, dot10, step_x)
    row1 = torch.lerp(dot01, dot11, step_x)
    noise = torch.lerp(row0, row1, step_y)

    return (
        noise.view(batch_size, bh, bw, gh, gw)
        .permute(0, 3, 1, 4, 2)
        .reshape(batch_size, gh * bh, gw * bw)
    )


def smooth_step(t):
    """
    Smooth step function.

    Args:
        t (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying smooth step.
    """
    return t * t * (3.0 - 2.0 * t)



class ZsnrVpredConditioningNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "zsnr": (["true", "false"], {"default": "true"}),
                "v_prediction": (["true", "false"], {"default": "true"}),
                "sigma_min": ("FLOAT", {
                    "default": 0.0292,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                }),
                "sigma_data": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1
                }),
                "min_snr_gamma": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "resolution_scaling": (["true", "false"], {"default": "true"})
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_conditioning"
    CATEGORY = "conditioning"

    def apply_conditioning(self, model, zsnr, v_prediction, sigma_min, sigma_data, min_snr_gamma, resolution_scaling):
        """
        Apply ZSNR and V-prediction conditioning to the model.

        Args:
            model: The diffusion model.
            zsnr (str): Whether to use ZSNR.
            v_prediction (str): Whether to use V-prediction.
            sigma_min (float): Minimum sigma value.
            sigma_data (float): Sigma data value.
            min_snr_gamma (float): Minimum SNR gamma.
            resolution_scaling (str): Whether to use resolution scaling.

        Returns:
            tuple: Tuple containing the updated model.
        """
        model_out = model.clone() if hasattr(model, 'clone') else model

        if not hasattr(model_out, "model_options"):
            model_out.model_options = {}

        # Update model options
        model_out.model_options.update({
            "v_prediction": v_prediction.lower() == "true",
            "sigma_data": sigma_data,
            "zsnr": zsnr.lower() == "true",
            "sigma_min": sigma_min,
            "min_snr_gamma": min_snr_gamma,  # Add MinSNR gamma
            "use_resolution_scaling": resolution_scaling.lower() == "true"  # Enable resolution scaling
        })

        return (model_out,)


class CFGRescaleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rescale_cfg": (["true", "false"], {"default": "true"}),
                "scale_method": ([
                    "karras", "standard", "exponential", "sgm_uniform",
                    "simple", "ddim_uniform", "normal"
                ], {"default": "karras"}),
                "rescale_multiplier": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05
                })
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_rescale"
    CATEGORY = "sampling"

    def apply_rescale(self, model, rescale_cfg, scale_method, rescale_multiplier):
        """
        Apply CFG rescaling to the model.

        Args:
            model: The diffusion model.
            rescale_cfg (str): Whether to rescale CFG.
            scale_method (str): Method to use for scaling.
            rescale_multiplier (float): Multiplier for scaling.

        Returns:
            tuple: Tuple containing the updated model.
        """
        model_out = model.clone() if hasattr(model, 'clone') else model

        if not hasattr(model_out, "model_options"):
            model_out.model_options = {}

        model_out.model_options.update({
            "rescale_cfg": rescale_cfg.lower() == "true",
            "scale_method": scale_method,
            "rescale_multiplier": rescale_multiplier
        })

        return (model_out,)


class LaplaceSchedulerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mu": ("FLOAT", {
                    "default": 0.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "beta": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                })
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_scheduler"
    CATEGORY = "sampling"

    def apply_scheduler(self, model, mu, beta):
        """
        Apply Laplace scheduler to the model.

        Args:
            model: The diffusion model.
            mu (float): Mu parameter for Laplace distribution.
            beta (float): Beta parameter for Laplace distribution.

        Returns:
            tuple: Tuple containing the updated model.
        """
        model_out = model.clone() if hasattr(model, 'clone') else model

        if not hasattr(model_out, "model_options"):
            model_out.model_options = {}

        model_out.model_options.update({
            "scheduler_name": "laplace",
            "scheduler_args": {
                "mu": mu,
                "beta": beta
            }
        })

        return (model_out,)

