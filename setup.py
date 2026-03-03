from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name = "cubicmultispline",
    version = "0.1.0",
    description = "Generation of cubic, multivariate splines from samples with arbitrary boundary conditions",
    package_dir={"": "src"},
    packages = find_packages(where="src"),
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license = "MIT License",
    author = "A. C. Feldkamp",
    author_email = "",
    url = "https://github.com/a118145/cubic-multivar-spline",
    classifiers =[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.8",
    install_requires =[
        "numpy>=2.4",
        "scipy>=1.12",
    ],
    extras_require={
        "dev": [
            "twine>=6.2",
            "setuptools>=82.0.0",
            "matplotlib>=3.10.0",
        ]
    },
)