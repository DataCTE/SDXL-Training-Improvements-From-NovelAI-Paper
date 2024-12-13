from setuptools import setup, find_packages
from src.utils.model import create_unet, create_vae

setup(
    name="sdxl-training-improvements",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "diffusers>=0.24.0",
        "transformers>=4.35.0",
        "bitsandbytes>=0.41.1",
        "tqdm",
        "Pillow",
        "sentencepiece",
        "accelerate>=0.24.0",
        "wandb",
        "huggingface-hub>=0.19.0",
        "basicsr",
        "facexlib",
        "realesrgan",
        "spacy>=3.8.0",
        "scikit-learn",
        "adamw_bf16",
        "filelock>=3.12.0",
        "einops>=0.6.1",
        "fairscale>=0.4.13",
        "fire>=0.5.0",
        "fsspec>=2023.6.0",
        "invisible-watermark>=0.2.0",
        "kornia>=0.6.9",
        "matplotlib>=3.7.2",
        "numpy>=1.24.4",
        "omegaconf>=2.3.0",
        "opencv-python>=4.6.0",
        "pandas>=2.0.3",
        "pytorch-lightning>=2.0.1",
        "pyyaml>=6.0.1",
        "scipy>=1.10.1",
        "tensorboardx>=2.6",
        "timm>=0.9.2",
        "webdataset>=0.2.33",
        "gradio"
    ],
    dependency_links=[
        "git+https://github.com/openai/CLIP.git#egg=clip"
    ]
) 