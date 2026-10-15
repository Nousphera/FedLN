"""Build the FedLN package and its native Flower application assets."""

from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="fedln", version="0.2.0",
    description="Labeling Chaos to Learning Harmony: Federated Learning with Noisy Labels",
    long_description=Path("README.md").read_text() if Path("README.md").exists() else "FedLN federated learning.",
    long_description_content_type="text/markdown",
    license="Apache-2.0", license_files=("LICENSE",), python_requires=">=3.11,<3.14",
    install_requires=[
        "torch==2.8.0", "torchvision==0.23.0", "flwr[simulation]==1.40.0",
        "numpy>=2,<3", "scipy>=1.14,<2", "scikit-learn>=1.6,<2",
        "cleanlab==2.9.0", "transformers==4.57.6", "timm==1.0.30", "pillow>=11,<13",
    ],
    package_dir={"": "src"}, packages=find_packages("src", include=["fedln", "fedln.*"]),
    data_files=[("share/fedln", ["setup.py", "LICENSE"]),
                ("share/fedln/scripts", [str(p) for p in Path("scripts").glob("*.py")])],
    entry_points={"console_scripts": ["fedln-experiment=fedln.experiment:main"]},
)
