import numpy
import pyopencl as cl
import pyopencl.cltypes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import os
import time


os.environ["PYOPENCL_NO_CACHE"] = "1"


def make_type(ctx, type_name, type_desc, device=None):
    """
    :return: CL code generated for given type and numpy.dtype instance
    """
    import pyopencl.tools
    dtype, cl_decl = cl.tools.match_dtype_to_c_struct(
        ctx.devices[0] if device is None else device, type_name, numpy.dtype(type_desc), context=ctx
    )
    type_def = cl.tools.get_or_register_dtype(type_name, dtype)
    return cl_decl, type_def


def copy_dev(ctx, buf):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=buf)


def alloc_like(ctx, buf):
    return cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=buf.nbytes)



PRNG_SOURCE = r"""
// source code:
// http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
uint MWC64X(uint2 *state);
uint MWC64X(uint2 *state) {
    enum { A = 4294883355U };
    uint x = (*state).x, c = (*state).y;  // Unpack the state
    uint res = x^c;                     // Calculate the result
    uint hi = mul_hi(x,A);              // Step the RNG
    x = x*A+c;
    c = hi+(x<c);
    *state = (uint2)(x,c);               // Pack the state back up
    return res;                        // Return the next result
}

void init_state(ulong seed, uint2* state);
void init_state(ulong seed, uint2* state) {
    int id = get_global_id(0) + 1;
    uint2 s = as_uint2(seed);
    (*state) = (uint2)(
        // create a mixture of id and two seeds
        (id + s.x & 0xFFFF) * s.y,
        (id ^ (s.y & 0xFFFF0000)) ^ s.x
    );
}

// retrieve random float in range [0.0; 1.0] (both inclusive)
inline float random(uint2* state) {
    return ((float)MWC64X(state)) / (float)0xffffffff;
}
"""
