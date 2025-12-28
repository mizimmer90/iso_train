"""Setup script for iso_train package."""

from setuptools import setup, find_packages

setup(
    name="iso_train",
    version="0.1.0",
    description="DDPM with Iso-Time Contrastive Learning",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
    ],
)

