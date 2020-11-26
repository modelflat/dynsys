import os

from datetime import datetime

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

version = (os.environ.get('VERSION') or datetime.utcnow().strftime("0.0.0.%Y%m%d%H%M%S")).lstrip('v')

setup(
    name="dynsys",
    version=version,

    description="",

    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/modelflat/dynsys",

    author="modelflat",

    classifiers=[
        "Development Status :: 2 - Pre-Alpha",

        "Programming Language :: Python :: 3",
    ],

    keywords="opencl research nonlinear dynamical-systems",

    project_urls={
        "Issues": "https://github.com/modelflat/dynsys/issues",
    },

    packages=find_packages(exclude=["examples"]),

    install_requires=["pyopencl", "PyQt5", "PySide2"],

    include_package_data=True,

    python_requires=">=3",
)
