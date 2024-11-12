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


class CustomKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": ([
                    "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
                    "dpmpp_2m", "dpmpp_2m_sde", "ddim", "uni_pc", "uni_pc_bh2"
                ],),
                "scheduler": ([
                    "karras", "exponential", "sgm_uniform", "simple",
                    "ddim_uniform", "normal"
                ],),
                "latent": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_type": ([
                    "gaussian", "uniform", "pyramid", "perlin", "laplacian"
                ], {"default": "gaussian"})
            },
            "optional": {
                "optional_positive": ("CONDITIONING",),
                "optional_negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def generate_pyramid_noise(self, shape, generator, device, discount=0.8):
        """
        Generate pyramid noise.

        Args:
            shape (tuple): Shape of the noise tensor (batch, channels, height, width).
            generator (torch.Generator): Random number generator.
            device (torch.device): Device to generate noise on.
            discount (float): Discount factor for each pyramid level.

        Returns:
            torch.Tensor: Pyramid noise tensor.
        """
        b, c, h, w = shape
        noise = torch.zeros(shape, device=device)
        for i in range(5):
            r = 2 ** i
            h_scaled = max(h // r, 1)
            w_scaled = max(w // r, 1)
            noise_level = torch.randn((b, c, h_scaled, w_scaled), generator=generator, device=device)
            noise += torch.nn.functional.interpolate(noise_level, size=(h, w), mode='nearest') * (discount ** i)
        return noise

    def generate_perlin_noise(self, shape, generator, device):
        """
        Generate Perlin noise.

        Args:
            shape (tuple): Shape of the noise tensor (batch, channels, height, width).
            generator (torch.Generator): Random number generator.
            device (torch.device): Device to generate noise on.

        Returns:
            torch.Tensor: Perlin noise tensor.
        """
        noise = torch.randn(shape, generator=generator, device=device) / 2.0
        noise_size_h = shape[2]
        noise_size_w = shape[3]
        batch_size = shape[0]

        # Generate perlin noise
        perlin = perlin_noise(
            grid_shape=(4, 4),  # Example grid shape; adjust as needed
            out_shape=(noise_size_h, noise_size_w),
            batch_size=batch_size,
            generator=generator
        ).to(device)

        # Add perlin noise to base noise
        noise += perlin.unsqueeze(1).expand(shape[0], -1, -1, -1)
        return noise / noise.std()

    def generate_laplacian_noise(self, shape, generator, device):
        """
        Generate Laplacian noise.

        Args:
            shape (tuple): Shape of the noise tensor (batch, channels, height, width).
            generator (torch.Generator): Random number generator.
            device (torch.device): Device to generate noise on.

        Returns:
            torch.Tensor: Laplacian noise tensor.
        """
        noise = torch.randn(shape, generator=generator, device=device) / 4.0
        laplace = Laplace(loc=0.0, scale=1.0)
        laplace_noise = laplace.sample(shape).to(device)
        noise += laplace_noise
        return noise / noise.std()

    def generate_noise(self, shape, seed, noise_type, device):
        """
        Generate noise based on the specified type.

        Args:
            shape (tuple): Shape of the noise tensor (batch, channels, height, width).
            seed (int): Seed for the random number generator.
            noise_type (str): Type of noise to generate.
            device (torch.device): Device to generate noise on.

        Returns:
            torch.Tensor: Generated noise tensor.
        """
        generator = torch.Generator(device=device).manual_seed(seed)

        if noise_type == "gaussian":
            noise = torch.randn(shape, generator=generator, device=device)
        elif noise_type == "uniform":
            noise = (torch.rand(shape, generator=generator, device=device) * 2 - 1) * 1.73
        elif noise_type == "pyramid":
            noise = self.generate_pyramid_noise(shape, generator, device)
        elif noise_type == "perlin":
            noise = self.generate_perlin_noise(shape, generator, device)
        elif noise_type == "laplacian":
            noise = self.generate_laplacian_noise(shape, generator, device)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        return noise.contiguous()

    def get_sigmas(self, model, steps, scheduler, denoise):
        """
        Get sigmas based on scheduler type.

        Args:
            model: The diffusion model.
            steps (int): Number of steps in the scheduler.
            scheduler (str): Scheduler type.
            denoise (float): Denoising strength.

        Returns:
            torch.Tensor: Scheduled sigma values.
        """
        total_steps = steps
        if denoise < 1.0:
            total_steps = max(int(steps / denoise), 1)  # Ensure at least 1 step

        # Get model options
        model_options = getattr(model, "model_options", {})
        use_scaling = model_options.get("use_resolution_scaling", False)
        sigma_min = model_options.get("sigma_min", 0.0292)
        zsnr = model_options.get("zsnr", False)

        # Get latent size from the model or latent tensor
        # Assuming the model has attributes to get spatial dimensions
        # Adjust this part based on your actual model structure
        try:
            latent_image = model.latent_image  # Placeholder attribute
            _, _, latent_height, latent_width = latent_image.shape
        except AttributeError:
            # Fallback: assume default scaling
            latent_height, latent_width = 64, 64  # Example defaults; adjust as needed

        height = latent_height * 8
        width = latent_width * 8

        # Calculate resolution-dependent sigma_max if using scaling
        if use_scaling and zsnr:
            base_res = 832 * 1216
            current_res = height * width
            scale_factor = (current_res / base_res) ** 0.5
            # Use 20000 as infinity approximation for ZTSNR
            sigma_max = 20000.0 * scale_factor
        else:
            sigma_max = 14.6146

        # Store values in model options for other components
        model_options["sigma_max"] = sigma_max
        model_options["sigma_min"] = sigma_min  # Store sigma_min as well

        # Scheduler selection
        if scheduler == "karras":
            sigmas = karras_scheduler(model, total_steps, sigma_min=sigma_min, sigma_max=sigma_max)
        elif scheduler == "exponential":
            sigmas = exponential_scheduler(model, total_steps, sigma_min=sigma_min, sigma_max=sigma_max)
        elif scheduler == "sgm_uniform":
            sigmas = sgm_uniform_scheduler(model, total_steps)
        elif scheduler == "simple":
            sigmas = simple_scheduler(model, total_steps)
        elif scheduler == "ddim_uniform":
            sigmas = ddim_uniform_scheduler(model, total_steps)
        elif scheduler == "normal":
            sigmas = normal_scheduler(model, total_steps, sigma_min=sigma_min, sigma_max=sigma_max)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        # Ensure positive strides for sigmas
        sigmas = sigmas[-(steps + 1):].contiguous()
        return sigmas

    def sample(self, model, noise_seed, steps, cfg, sampler_name, scheduler, latent, denoise,
           noise_type, optional_positive=None, optional_negative=None):
        """
        Main sampling function.
        """
        # Get model options
        model_options = getattr(model, "model_options", {})
        rescale_cfg = model_options.get("rescale_cfg", True)
        scale_method = model_options.get("scale_method", "karras") 
        rescale_multiplier = model_options.get("rescale_multiplier", 0.7)

        # Setup conditions
        positive = optional_positive if optional_positive is not None else {"samples": []}
        negative = optional_negative if optional_negative is not None else {"samples": []}
        
        # Get device - Fixed to get from first parameter
        device = next(model.model.diffusion_model.parameters()).device
        
        # Setup latent
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        # Get sigmas
        sigmas = self.get_sigmas(model, steps, scheduler, denoise)
        
        # Generate noise only once at the start
        noise = self.generate_noise(latent_image.shape, noise_seed, noise_type, device)
        noise = noise.contiguous() # Ensure contiguous memory layout
        
        # Rescale CFG based on method
        if rescale_cfg:
            match scale_method:
                case "normal":
                    # Get sigma ratio with epsilon to prevent division by zero
                    eps = 1e-7
                    sigma_ratio = sigmas[0] / (sigmas[-2] + eps)
                    
                    # Clip ratio to reasonable range
                    sigma_ratio = torch.clamp(sigma_ratio, 1e-4, 1e4)
                    
                    # Calculate normal CDF scaling on device
                    x = sigma_ratio / math.sqrt(2)
                    x = x.to(sigmas.device, dtype=sigmas.dtype)
                    
                    # Calculate scale with bounds
                    scale = torch.clamp(0.5 * (1 + torch.erf(x)), 0.1, 10.0)
                    
                    # Apply rescaling with safety check
                    cfg = cfg * rescale_multiplier * scale.item()
                    
                case "karras":
                    # Karras scaling from EDM paper with safety bounds
                    scale = torch.clamp(sigmas[0] / (sigmas[-2] + 1e-7), 0.1, 10.0)
                    cfg = cfg * rescale_multiplier * scale
                case "exponential":
                    # Exponential scaling
                    cfg = cfg * rescale_multiplier * torch.exp(-sigmas[-2] / sigmas[0])
                case "sgm_uniform":
                    # Linear scaling based on step progress
                    cfg = cfg * rescale_multiplier * (1.0 - steps / len(sigmas))
                case "simple":
                    # Simple linear scaling
                    cfg = cfg * rescale_multiplier
                case "ddim_uniform":
                    # DDIM-style uniform scaling
                    cfg = cfg * rescale_multiplier * torch.sqrt(sigmas[0] / sigmas[-2])

        # Ensure cfg is finite and reasonable
        cfg = torch.clamp(torch.tensor(cfg), 1.0, 100.0).item()

        # Sample without additional noise in the loop
        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, sigmas, positive, negative,
            latent_image, denoise=denoise
        )

        out = latent.copy()
        out["samples"] = samples
        return (out,)


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

