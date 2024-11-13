import logging
import torch
from PIL import Image, ImageDraw
import math
from diffusers import StableDiffusionPipeline, DDIMScheduler
from enum import Enum

MAX_RESOLUTION = 8192

class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class UltimateSDUpscaler:
    def __init__(self, model_path="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device
        
        # Initialize pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        ).to(device)
        
        # Use DDIM scheduler for better quality
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        
        # Enable memory efficient attention if available
        if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
            self.pipeline.enable_xformers_memory_efficient_attention()

    def upscale(
        self,
        image,
        prompt,
        negative_prompt="",
        upscale_by=2.0,
        seed=None,
        steps=20,
        cfg=7.5,
        denoise=0.2,
        tile_width=512,
        tile_height=512,
        tile_padding=32,
        mode_type=USDUMode.LINEAR,
        mask_blur=8,
        seam_fix_mode=USDUSFMode.NONE,
        seam_fix_denoise=0.35,
        seam_fix_width=64,
        seam_fix_mask_blur=8,
        seam_fix_padding=16,
        force_uniform_tiles=True
    ):
        """Main upscaling method"""
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            
        # Calculate target size
        target_width = int(image.width * upscale_by)
        target_height = int(image.height * upscale_by)
        
        # First upscale the image using basic method
        upscaled = image.resize((target_width, target_height), Image.LANCZOS)
        
        # If no redraw mode selected, return basic upscale
        if mode_type == USDUMode.NONE:
            return upscaled
            
        # Process tiles
        tiles, masks, positions = self._split_into_tiles(
            upscaled, 
            tile_width, 
            tile_height, 
            tile_padding,
            mode_type
        )
        
        # Process each tile
        processed_tiles = []
        for tile, mask, pos in zip(tiles, masks, positions):
            processed_tile = self._process_tile(
                tile, mask, prompt, negative_prompt, steps, cfg, denoise
            )
            processed_tiles.append(processed_tile)
        
        # Merge tiles
        merged = self._merge_tiles(processed_tiles, masks, positions, (target_width, target_height))
        
        # Fix seams if requested
        if seam_fix_mode != USDUSFMode.NONE:
            merged = self._fix_seams(
                merged,
                prompt,
                negative_prompt,
                steps,
                cfg,
                seam_fix_mode,
                seam_fix_denoise,
                seam_fix_width,
                seam_fix_mask_blur,
                seam_fix_padding
            )
        
        return merged

    def _split_into_tiles(self, image, tile_width, tile_height, padding, mode):
        """Split image into tiles for processing"""
        tiles = []
        masks = []
        positions = []
        
        width, height = image.size
        rows = math.ceil(height / tile_height)
        cols = math.ceil(width / tile_width)
        
        for row in range(rows):
            for col in range(cols):
                # Calculate tile position
                x = col * tile_width
                y = row * tile_height
                
                # Create tile mask based on mode
                if mode == USDUMode.LINEAR:
                    mask = self._create_linear_mask(tile_width, tile_height)
                elif mode == USDUMode.CHESS:
                    mask = self._create_chess_mask(tile_width, tile_height, row, col)
                
                # Extract and pad tile
                tile = self._extract_tile(
                    image, x, y, tile_width, tile_height, padding
                )
                
                tiles.append(tile)
                masks.append(mask)
                positions.append((x, y))
                
        return tiles, masks, positions

    def _create_linear_mask(self, width, height):
        """Create linear gradient mask"""
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for y in range(height):
            alpha = int(255 * (y / height))
            draw.line([(0, y), (width, y)], fill=alpha)
        return mask

    def _create_chess_mask(self, width, height, row, col):
        """Create checkerboard mask"""
        mask = Image.new('L', (width, height), 255 if (row + col) % 2 == 0 else 0)
        return mask

    def _process_tile(self, tile, mask, prompt, negative_prompt, steps, cfg, denoise):
        """Process a single tile using the diffusion model"""
        # Prepare the conditioning image (tile)
        conditioning_image = tile.copy()
        
        # Create inpainting mask
        inpaint_mask = mask.convert('L')
        
        # Run the pipeline
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=conditioning_image,
            mask_image=inpaint_mask,
            num_inference_steps=steps,
            guidance_scale=cfg,
            strength=denoise,
        ).images[0]
        
        return output

    def _merge_tiles(self, tiles, masks, positions, target_size):
        """Merge processed tiles back into final image"""
        final_image = Image.new('RGB', target_size)
        
        for tile, mask, (x, y) in zip(tiles, masks, positions):
            # Create a mask for blending
            blend_mask = mask.resize(tile.size, Image.LANCZOS)
            
            # Paste tile using mask
            final_image.paste(
                tile,
                (x, y),
                mask=blend_mask
            )
            
        return final_image

    def _fix_seams(
        self,
        image,
        prompt,
        negative_prompt,
        steps,
        cfg,
        mode,
        denoise,
        width,
        mask_blur,
        padding
    ):
        """Fix seams between tiles"""
        if mode == USDUSFMode.NONE:
            return image
            
        if mode == USDUSFMode.BAND_PASS:
            return self._band_pass_fix(
                image, prompt, negative_prompt,
                steps, cfg, denoise, width, mask_blur, padding
            )
        elif mode == USDUSFMode.HALF_TILE:
            return self._half_tile_fix(
                image, prompt, negative_prompt,
                steps, cfg, denoise, width, mask_blur, padding
            )
        elif mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            image = self._half_tile_fix(
                image, prompt, negative_prompt,
                steps, cfg, denoise, width, mask_blur, padding
            )
            return self._intersection_fix(
                image, prompt, negative_prompt,
                steps, cfg, denoise, width, mask_blur, padding
            )
