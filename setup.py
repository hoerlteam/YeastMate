import setuptools
from yeastmatedetector import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yeastmatedetector",
    version=__version__,
    author="David Bunk",
    author_email="bunk@bio.lmu.de",
    description="Detector module for YeastMate.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hoerlteam/YeastMateDetector",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.9',
    install_requires=[
        "scikit-image",
        "imgaug",
        "numpy",
        'scipy',
    ]
)
