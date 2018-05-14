from setuptools import setup, find_packages

with open("README.rst", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dynsys",
    version="0.1.2",

    description="Small collection of tools to perform basic modeling of dynamical systems",

    long_description=long_description,
    long_description_content_type="text/x-rst",

    url="https://bitbucket.org/modelflat/dynsys",

    author="Andrew Elistratov",
    author_email="zhykreg@gmail.com",

    license="MIT",

    classifiers=[
        "Development Status :: 3 - Alpha",

        "Intended Audience :: Science/Research",

         "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3",

        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],

    keywords="opencl research nonlinear dynamical-systems",

    project_urls={
        "Issues": "https://bitbucket.org/modelflat/dynsys/issues",
    },

    packages=find_packages(exclude=["examples"]),

    install_requires=["numpy", "pyopencl", "PyQt5"],

    python_requires=">=3",


)