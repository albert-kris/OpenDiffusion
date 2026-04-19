from setuptools import setup, find_packages

setup(
    name="zhou_diffusion",
    version="0.1.0",
    description="OpenDiffusion: A foundational diffusion model codebase",
    author="haidong Hu",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "einops",
    ],
)