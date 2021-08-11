import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yeastmatedetector",
    version="0.11.0",
    author="David Bunk",
    author_email="bunk@bio.lmu.de",
    description="Detector module for YeastMate.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CALM-LMU/YeastMateDetector",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "scikit-image",
        "imgaug",
        "numpy",
        'scipy',
    ]
)