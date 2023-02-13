from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent

setup(
    name="housing-inference",
    version="0.1.0",
    packages=find_packages("."),
)
