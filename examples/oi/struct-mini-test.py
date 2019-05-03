import numpy
import pyopencl as cl

src = r"""

void do_step(system_t* d, system_t* r, struct params_t* p) {
    // code 
}

kernel void test(
    global system_t* s
) {
    const int id = get_global_id(0);
    printf("s.a = %f\n", s[id].a);
}

"""


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


ctx = cl.create_some_context(answers=[0, 0])
queue = cl.CommandQueue(ctx)

type_src, type_def = make_type(
    ctx=ctx,
    type_name="system_t",
    type_desc=[
        ("a", numpy.float32),
        ("b", numpy.float32),
    ]
)

prg = cl.Program(ctx, type_src + src).build()

s = numpy.empty((10,), dtype=type_def)

s["a"] = range(10)

s_dev = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=s)

prg.test(
    queue, (10,), None, s_dev
)