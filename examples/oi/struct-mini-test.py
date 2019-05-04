import numpy
import pyopencl as cl

src = r"""

kernel void test(
    global system_t* s_
) {
    const int id = get_global_id(0);
    system_t s = s_[id];
    printf("s = {\n\t{\n\t\tx: %f,\n\t\ty: %f\n\t},\n\ta: %f,\n\tb: %f\n}\n",
        s.v.x, s.v.y, s.a, s.b 
    );
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

val_t_src, val_t = make_type(
    ctx=ctx,
    type_name="val_t",
    type_desc=[
        ("x", numpy.float64),
        ("y", numpy.float64)
    ]
)

system_t_src, system_t = make_type(
    ctx=ctx,
    type_name="system_t",
    type_desc=[
        ("v", val_t),
        ("a", numpy.float64),
        ("b", numpy.float64),
    ]
)

prg = cl.Program(ctx, val_t_src + system_t_src + src).build()

s = numpy.zeros((10,), dtype=system_t)

s["v"]["x"] = range(10)
s["v"]["y"] = range(9, -1, -1)


s_dev = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=s)

prg.test(
    queue, (10,), None, s_dev
)