import math
import torch
import logging
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from enum import Enum


logger = logging.getLogger(__name__)

class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class UltimateUpscaler:
    def __init__(
        self,
        model_path: str = "Lykon/DreamShaper",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """Initialize the Ultimate Upscaler with DreamShaper model"""
        self.device = device
        self.dtype = dtype
        
        logger.info(f"Loading DreamShaper model from {model_path}")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        # Use DDIM scheduler for better quality
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Enable memory optimizations
        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()

    def _create_tile_mask(
        self,
        width: int,
        height: int,
        x: int,
        y: int,
        tile_width: int,
        tile_height: int,
        padding: int = 32
    ) -> Image.Image:
        """Create a mask for a single tile"""
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        
        x1 = x * tile_width
        y1 = y * tile_height
        x2 = min(x1 + tile_width, width)
        y2 = min(y1 + tile_height, height)
        
        draw.rectangle([x1, y1, x2, y2], fill="white")
        return mask

    def _process_tile(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_steps: int = 20,
        guidance_scale: float = 7.5,
        strength: float = 0.4
    ) -> Image.Image:
        """Process a single tile using the pipeline"""
        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                width=image.width,
                height=image.height
            ).images[0]
        
        return output

    def _create_chess_pattern(
        self,
        rows: int,
        cols: int
    ) -> List[List[bool]]:
        """Create a chess pattern for tile processing"""
        pattern = []
        for y in range(rows):
            row = []
            for x in range(cols):
                color = (x + y) % 2 == 0
                row.append(color)
            pattern.append(row)
        return pattern

    def _process_linear(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        tile_width: int,
        tile_height: int,
        padding: int,
        num_steps: int,
        guidance_scale: float,
        strength: float
    ) -> Image.Image:
        """Process image using linear tile pattern"""
        width, height = image.size
        rows = math.ceil(height / tile_height)
        cols = math.ceil(width / tile_width)
        
        result = image.copy()
        
        for y in range(rows):
            for x in range(cols):
                mask = self._create_tile_mask(width, height, x, y, tile_width, tile_height, padding)
                result = self._process_tile(
                    result, mask, prompt, negative_prompt,
                    num_steps, guidance_scale, strength
                )
        
        return result

    def _process_chess(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        tile_width: int,
        tile_height: int,
        padding: int,
        num_steps: int,
        guidance_scale: float,
        strength: float
    ) -> Image.Image:
        """Process image using chess tile pattern"""
        width, height = image.size
        rows = math.ceil(height / tile_height)
        cols = math.ceil(width / tile_width)
        
        pattern = self._create_chess_pattern(rows, cols)
        result = image.copy()
        
        # Process white tiles
        for y in range(rows):
            for x in range(cols):
                if pattern[y][x]:
                    mask = self._create_tile_mask(width, height, x, y, tile_width, tile_height, padding)
                    result = self._process_tile(
                        result, mask, prompt, negative_prompt,
                        num_steps, guidance_scale, strength
                    )
        
        # Process black tiles
        for y in range(rows):
            for x in range(cols):
                if not pattern[y][x]:
                    mask = self._create_tile_mask(width, height, x, y, tile_width, tile_height, padding)
                    result = self._process_tile(
                        result, mask, prompt, negative_prompt,
                        num_steps, guidance_scale, strength
                    )
        
        return result

    def _fix_seams(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        guidance_scale: float,
        mode: USDUSFMode,
        denoise: float,
        width: int,
        mask_blur: int,
        padding: int
    ) -> Image.Image:
        """Fix seams between tiles"""
        if mode == USDUSFMode.NONE:
            return image
            
        result = image.copy()
        img_width, img_height = image.size
        
        if mode == USDUSFMode.BAND_PASS:
            # Create vertical and horizontal bands
            for x in range(1, math.ceil(img_width / width)):
                mask = Image.new("L", (img_width, img_height), "black")
                draw = ImageDraw.Draw(mask)
                x_pos = x * width - width // 2
                draw.rectangle([x_pos, 0, x_pos + width, img_height], fill="white")
                if mask_blur > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(mask_blur))
                result = self._process_tile(
                    result, mask, prompt, negative_prompt,
                    num_steps, guidance_scale, denoise
                )
            
            for y in range(1, math.ceil(img_height / width)):
                mask = Image.new("L", (img_width, img_height), "black")
                draw = ImageDraw.Draw(mask)
                y_pos = y * width - width // 2
                draw.rectangle([0, y_pos, img_width, y_pos + width], fill="white")
                if mask_blur > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(mask_blur))
                result = self._process_tile(
                    result, mask, prompt, negative_prompt,
                    num_steps, guidance_scale, denoise
                )
                
        elif mode in [USDUSFMode.HALF_TILE, USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS]:
            # Process half-tile overlaps
            gradient = Image.linear_gradient("L")
            
            # Process vertical seams
            for x in range(math.ceil(img_width / width) - 1):
                mask = Image.new("L", (img_width, img_height), "black")
                x_pos = x * width + width // 2
                gradient_tile = gradient.resize((width, img_height))
                mask.paste(gradient_tile, (x_pos, 0))
                if mask_blur > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(mask_blur))
                result = self._process_tile(
                    result, mask, prompt, negative_prompt,
                    num_steps, guidance_scale, denoise
                )
            
            # Process horizontal seams
            for y in range(math.ceil(img_height / width) - 1):
                mask = Image.new("L", (img_width, img_height), "black")
                y_pos = y * width + width // 2
                gradient_tile = gradient.rotate(90).resize((img_width, width))
                mask.paste(gradient_tile, (0, y_pos))
                if mask_blur > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(mask_blur))
                result = self._process_tile(
                    result, mask, prompt, negative_prompt,
                    num_steps, guidance_scale, denoise
                )
            
            # Process intersections if requested
            if mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
                for x in range(math.ceil(img_width / width) - 1):
                    for y in range(math.ceil(img_height / width) - 1):
                        mask = Image.new("L", (img_width, img_height), "black")
                        x_pos = x * width + width // 2
                        y_pos = y * width + width // 2
                        gradient_tile = Image.radial_gradient("L").resize((width, width))
                        mask.paste(gradient_tile, (x_pos, y_pos))
                        if mask_blur > 0:
                            mask = mask.filter(ImageFilter.GaussianBlur(mask_blur))
                        result = self._process_tile(
                            result, mask, prompt, negative_prompt,
                            num_steps, guidance_scale, denoise
                        )
        
        return result

    def upscale(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        upscale_factor: float = 2.0,
        tile_width: int = 512,
        tile_height: int = 512,
        padding: int = 32,
        num_steps: int = 20,
        guidance_scale: float = 7.5,
        strength: float = 0.4,
        mode: USDUMode = USDUMode.LINEAR,
        seam_fix_mode: USDUSFMode = USDUSFMode.HALF_TILE,
        seam_fix_denoise: float = 0.35,
        seam_fix_width: int = 64,
        seam_fix_mask_blur: int = 8,
        seam_fix_padding: int = 16,
    ) -> Image.Image:
        """
        Upscale an image using the Ultimate SD Upscaler approach
        """
        # Calculate target size
        target_width = int(image.width * upscale_factor)
        target_height = int(image.height * upscale_factor)
        
        # Initial upscale
        logger.info(f"Initial upscale from {image.size} to {target_width}x{target_height}")
        upscaled = image.resize((target_width, target_height), Image.LANCZOS)
        
        if mode == USDUMode.NONE:
            return upscaled
            
        # Process tiles
        logger.info(f"Processing tiles using {mode.name} mode")
        if mode == USDUMode.LINEAR:
            processed = self._process_linear(
                upscaled, prompt, negative_prompt,
                tile_width, tile_height, padding,
                num_steps, guidance_scale, strength
            )
        else:  # CHESS mode
            processed = self._process_chess(
                upscaled, prompt, negative_prompt,
                tile_width, tile_height, padding,
                num_steps, guidance_scale, strength
            )
            
        # Fix seams if requested
        if seam_fix_mode != USDUSFMode.NONE:
            logger.info(f"Fixing seams using {seam_fix_mode.name} mode")
            processed = self._fix_seams(
                processed, prompt, negative_prompt,
                num_steps, guidance_scale, seam_fix_mode,
                seam_fix_denoise, seam_fix_width,
                seam_fix_mask_blur, seam_fix_padding
            )
            
        return processed 

    def _create_linear_gradient(self, mode="L", size=(512, 512)) -> Image.Image:
        """Create a linear gradient from black to white"""
        gradient = Image.new(mode, size)
        draw = ImageDraw.Draw(gradient)
        
        for x in range(size[0]):
            # Calculate color value based on x position
            color = int(255 * (x / size[0]))
            draw.line([(x, 0), (x, size[1])], fill=color)
        
        return gradient

    def _create_radial_gradient(self, mode="L", size=(512, 512)) -> Image.Image:
        """Create a radial gradient from white center to black edges"""
        gradient = Image.new(mode, size)
        draw = ImageDraw.Draw(gradient)
        
        center = (size[0] // 2, size[1] // 2)
        max_radius = math.sqrt(center[0]**2 + center[1]**2)
        
        for x in range(size[0]):
            for y in range(size[1]):
                # Calculate distance from center
                distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
                # Calculate color value based on distance
                color = int(255 * (1 - min(1, distance / max_radius)))
                draw.point((x, y), fill=color)
        
        return gradient 

    def upscale_batch(
        self,
        images: List[Image.Image],
        prompt: str,
        negative_prompt: str = "",
        **kwargs
    ) -> List[Image.Image]:
        """
        Upscale a batch of images using the same settings
        """
        results = []
        for image in images:
            try:
                result = self.upscale(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image: {str(e)}")
                # Append original image if processing fails
                results.append(image)
        
        return results