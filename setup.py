from setuptools import setup, find_packages

setup(
    name="dynsys",
    version="0.1.0",

    description="Small collection of tools to perform basic modeling of dynamical systems",

    long_description="""
    """,
    long_description_content_type="text/markdown",

    url="https://bitbucket.org/modelflat/dynsys",

    author="Andrew Elistratov",
    author_email="zhykreg@gmail.com",

    license="MIT",

    classifiers=[
        "Development Status :: 3 - Alpha",

        "Topic :: Software Development :: Build Tools",

        "Intended Audience :: Science/Research",

         "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3",
    ],

    keywords="opencl research nonlinear dynamical-systems",

    project_urls={
        "Source": "https://bitbucket.org/modelflat/dynsys/src",
        "Issues": "https://bitbucket.org/modelflat/dynsys/issues",
    },

    packages=find_packages(exclude=["examples"]),

    install_requires=["numpy", "pyopencl", "PyQt4"],

    python_requires=">=3",


)