import numpy
import pyopencl as cl
import pyopencl.cltypes
from dynsys import SimpleApp


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


PRNG_SOURCE = r"""
// source code:
// http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
uint MWC64X(uint2 *state);
uint MWC64X(uint2 *state) {
    enum{ A=4294883355U };
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


SERIES_SOURCE = """

#define _F(x, y, a) (1 - a*x*x + y)

// HENON MAP hardcoded
void do_step(system_t* d_, system_t* r_, param_t* p) {
    system_t d = *d_;
    system_t r = *r_;
    
    (*d_).x = _F(d.x, d.y, d.a);
    (*d_).y = d.b * d.x + (*p).D * random(&((*p).rng_state));
    
    (*r_).x = _F(r.x, r.y, r.a) + (*p).eps * (_F(d.x, d.y, d.a) - _F(r.x, r.y, r.a));
    (*r_).y = r.b * r.y + (*p).D * random(&((*p).rng_state));
}

kernel compute_time_series(
    global system_t* d_systems,
    global system_t* r_systems,
    global param_t* params,
    
    int skip, 
    int iter,
    
    global double* drive,
    global double* response
) {
    const int id = get_global_id(0);
    system_t d = d_systems[id];
    system_t r = r_systems[id];
    param_t p = params[id];
    
    for (int i = 0; i < skip; ++i) {
        do_step(&d, &r, &p);
    }
    
    drive += 2 * id * iter
    //       ^ dim
    response += 2 * id * iter
    //          ^ dim
    
    for (int i = 0; i < iter; ++i) {
        do_step(&d, &r, &p);
        // save drive
        drive[2*i + 0] = d.x;
        drive[2*i + 1] = d.y;
        // save response
        response[2*i + 0] = r.x;
        response[2*i + 1] = r.y;
    }
}

"""


class SystemWithNoise:

    def __init__(self, ctx):
        self.ctx = ctx

        system_t_src, self.system_t = make_type(ctx, "system_t", [
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("a", numpy.float64),
            ("b", numpy.float64),
        ])
        param_t_src, self.param_t = make_type(ctx, "param_t", [
            ("eps", numpy.float64),
            ("D", numpy.float64),
            ("rng_state", cl.cltypes.uint2),
        ])
        sources = [
            PRNG_SOURCE, system_t_src, param_t_src, SERIES_SOURCE
        ]
        self.prg = cl.Program(ctx, "\n".join(sources)).build()

    def compute(self, queue, drive: dict, response: dict, eps: float, D: float, skip, iter):
        num_par_systems = 1  # should be locked to 1 for now

        drive_sys = numpy.empty(num_par_systems, dtype=self.system_t)
        response_sys = numpy.empty(num_par_systems, dtype=self.system_t)

        for i in range(num_par_systems):
            for k, v in drive.items():
                drive_sys[i][k] = v

            for k, v in response.items():
                response_sys[i][k] = v

        drive_sys_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=drive_sys)
        response_sys_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=response_sys)

        params = numpy.empty(num_par_systems, dtype=self.param_t)
        params["eps"] = eps
        params["D"] = D
        params["rng_state"] = numpy.random.randint(0, size=2, dtype=numpy.uint32)
        params_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=params)

        drive = numpy.empty((iter, 2), dtype=numpy.float64)
        drive_dev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=drive.nbytes)

        response = numpy.empty((iter, 2), dtype=numpy.float64)
        response_dev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=response.nbytes)

        self.prg.compute_time_series(
            queue, (num_par_systems,), None,
            drive_sys_dev,
            response_sys_dev,
            params_dev,
            numpy.int32(skip), numpy.int32(iter),
            drive_dev,
            response_dev
        )

        cl.enqueue_copy(queue, drive, drive_dev)
        cl.enqueue_copy(queue, response, response_dev)

        return drive, response


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("6.9")




if __name__ == '__main__':
    App().run()