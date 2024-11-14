# Make some patches to the script
from .ultimate_upscaler import UltimateUpscaler as usdu
import math
from PIL import Image


if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image

#
# Instead of using multiples of 64, use multiples of 8
#

def round_length(length, multiple=8):
    return math.ceil(length / multiple) * multiple


# Upscaler
old_init = usdu.UltimateUpscaler.__init__

def new_init(self, *args, **kwargs):
    # Extract image and upscale factor before calling old_init
    if 'image' in kwargs:
        image = kwargs['image']
    elif args:
        image = args[0]  # Assuming image is first positional argument
    else:
        raise ValueError("Image not provided to initializer")

    if 'p' in kwargs:
        p = kwargs['p']
    elif len(args) > 1:
        p = args[1]  # Assuming p is second positional argument
    else:
        raise ValueError("Processing parameters not provided to initializer")

    # Round dimensions before original initialization
    p.width = round_length(image.width * p.upscale_by)
    p.height = round_length(image.height * p.upscale_by)

    # Now call original init with updated parameters
    old_init(self, *args, **kwargs)

usdu.UltimateUpscaler.__init__ = new_init

# Redraw
old_setup_redraw = usdu.USDURedraw.init_draw

def new_setup_redraw(self, p, width, height):
    mask, draw = old_setup_redraw(self, p, width, height)
    p.width = round_length(self.tile_width + self.padding)
    p.height = round_length(self.tile_height + self.padding)
    return mask, draw

usdu.USDURedraw.init_draw = new_setup_redraw

# Seams fix
old_setup_seams_fix = usdu.USDUSeamsFix.init_draw

def new_setup_seams_fix(self, p):
    old_setup_seams_fix(self, p)
    p.width = round_length(self.tile_width + self.padding)
    p.height = round_length(self.tile_height + self.padding)

usdu.USDUSeamsFix.init_draw = new_setup_seams_fix