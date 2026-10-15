from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="structure-loss-classification",
    version="1.0",
    description="Research workflows for structure loss classification in silicon ingots",
    long_description=Path(__file__).with_name("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/alfredoisg/structure-loss-classification",
    license="MIT",
    packages=find_packages(),
)
