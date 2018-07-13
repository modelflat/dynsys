# dynsys
[![GitHub version](https://badge.fury.io/gh/modelflat%2Fdynsys.svg)](https://badge.fury.io/gh/modelflat%2Fdynsys)
[![PyPi version](https://pypip.in/v/dynsys/badge.png)](https://crate.io/packages/dynsys/)

This library contains a collection of utility classes and functions for basic dynamical system 
modeling tasks (such as computing a parameter maps, cobweb (aka Verhulst) diagrams or basins of attraction).

Computations are accelerated using [OpenCL](https://www.khronos.org/opencl). PyQt-based UI elements are provided for displaying
and interacting with data. 

Warning for potential users! Only kind of "documentation" currently available is (poorly commented) examples, 
located under the "examples/" directory. There are plans on writing full-featured documentation across project,
but these are in the not-so-near-future.

### Probably useful links
 
 1. [PyOpenCL](https://github.com/inducer/pyopencl) -- OpenCL integration for Python
 2. [PyQt5](https://pypi.org/project/PyQt5/) -- Python bindings for Qt platform toolkit
 3. [Pre-built Python packages repo by Christoph Gohlke](https://www.lfd.uci.edu/~gohlke/pythonlibs/) -- for those who have troubles installing Python packages under Windows
