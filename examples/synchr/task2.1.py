import numpy
import pyopencl as cl


SOURCE = """

typedef float real;

struct vars_t {
    real x, y, z;
};

struct params_t {
    real a, b, p;
};


kernel void compute_points(
    vars_t vars,
    const param_t params,
    
)


"""

