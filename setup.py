from setuptools import setup, find_packages

setup(
    name="m2lrf",
    version="1.0.0",
    description="2-Bit Dual-Basis Quantization & Fine-Tuning Engine with LoftQ SVD Initialization",
    author="MD-Mushfiqur Rahim (M) & Agent L",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.26.0",
        "scipy>=1.10.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
