import torch
from torchvision import transforms

def ensure_three_channels(x):
    return x[:3]

def convert_to_bfloat16(x):
    return x.to(torch.bfloat16)

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        ensure_three_channels,
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        convert_to_bfloat16
    ]) 