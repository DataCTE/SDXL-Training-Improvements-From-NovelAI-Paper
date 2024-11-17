from setuptools import setup, find_packages

setup(
    name="sdxl-training-improvements",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "diffusers>=0.18.0",
        "bitsandbytes>=0.41.0",
    ],
)
