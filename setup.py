"""Setup module."""

import os
from pathlib import Path

from setuptools import find_packages, setup


def parse_requirements_file():
    """Parse requirements.txt style files."""
    path = os.path.join(Path(__file__).parent, "requirements.txt")
    reqs = []
    with open(path) as file:
        for line in file:
            reqs.append(line.strip("\n"))

    return reqs


setup(
    name="pydata_sktime",
    version="0.0",
    packages=find_packages(),
    description="PyData Seattle 2023 sktime workshop",
    author="sktime",
    # requirements from the requiremnts.txt file
    install_requires=parse_requirements_file(),
)
