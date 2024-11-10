import torch
import contextlib

@contextlib.contextmanager
def to_device(model, device):
    """Temporarily move model to device"""
    original_device = next(model.parameters()).device
    try:
        model.to(device)
        yield model
    finally:
        model.to(original_device) 