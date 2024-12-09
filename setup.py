from setuptools import setup, find_packages

setup(
    name="stylegan_project",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "pillow>=8.0.0",
        "matplotlib>=3.4.0",
        "fastapi>=0.78.0",
        "uvicorn>=0.18.0",
        "pytest>=7.0.0",
    ],
)
"""from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stylegan-project",  # Replace with the name of your project
    version="0.1.0",  # Versioning follows Semantic Versioning
    author="Your Name",
    author_email="your_email@example.com",
    description="A GAN-based model to generate images of faces using StyleGAN.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stylegan-project",
    packages=find_packages(exclude=["tests", "data", "notebooks"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "pillow>=8.0.0",
        "matplotlib>=3.4.0",
        "fastapi>=0.78.0",
        "uvicorn>=0.18.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "mypy",
            "sphinx",
        ],
    },
    entry_points={
        "console_scripts": [
            "stylegan-train=scripts.train:main",
            "stylegan-generate=scripts.generate:main",
        ],
    },
)"""