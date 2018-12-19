import numpy
from dynsys import SimpleApp


import pyopencl as cl

from dynsys.LCE import dummyOption


LOSI = """
float2 forward_fun(
    float2 p,
    float a, float b
) {
    float2 res;
    res.x = 1 - a*fabs(p.x) + b*p.y;
    res.y = p.x;
    return res;
}

float2 backward_fun(
    float2 p,
    float a, float b
) {
    float2 res;
    res.x = p.y;
    res.y = (p.x - 1.0 + a*(res.x))/b;
    return res;
}
"""


DEFS = r"""
#define real float
#define real2 float2
"""


HOMOCLINIC_SOURCE = r"""
kernel void homoclinicPlot(
    const real a,
    const real b,
    const int count,
    global const real2* initial_f, 
    global const real2* initial_b,
    global real2* forw,
    global real2* back,
) {
    int idx = get_global_id(0);
    forw += idx*count;
    back += idx*count;
    
    real2 fp = initial_f[idx];
    real2 bp = initial_b[idx];
    
    for(int i = 0; idx < count; ++idx){
        forw[i] = fp;        
        fp = forward_fun(fp, a, b);
        back[i] = bp;
        bp = backward_fun(bp, a, b);
    }
}
"""


def generatePointsInEps(stablePoint: tuple, eps: tuple, pointCount: int, dtype=numpy.float32):
    return numpy.array((
        numpy.random.uniform(
            low=stablePoint[0] - eps[0],
            high=stablePoint[1] + eps[1],
            size=pointCount
        ),
        numpy.random.uniform(
            low=stablePoint[0] - eps[0],
            high=stablePoint[1] + eps[1],
            size=pointCount
        )
    ), dtype=dtype).T


class Homoclinic:
    
    def __init__(self, ctx, queue, fn):
        self.ctx, self.queue = ctx, queue
        self.prg = cl.Program(self.ctx, "\n".join(
            (DEFS, fn, HOMOCLINIC_SOURCE)
        )).build(options=[dummyOption()])
        self.type = numpy.float32
    
    def __call__(self,
                 stablePoint: tuple, eps: float, pointCount: int, iterations: int,
                 a: float, b: float, plot_f, plot_b):

        spfHost = numpy.array((
            numpy.random.uniform(low=stablePoint[0] - eps / 2, high=stablePoint[0] + eps / 2, size=pointCount),
            numpy.random.uniform(low=stablePoint[1] - eps / 2, high=stablePoint[1] + eps / 2, size=pointCount)
        ), dtype=self.type).T
        spfDev = cl.Buffer(self.ctx,
                           flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=spfHost)
        spbHost = numpy.array((
            numpy.random.uniform(low=stablePoint[0] - eps / 2, high=stablePoint[0] + eps / 2, size=pointCount),
            numpy.random.uniform(low=stablePoint[1] - eps / 2, high=stablePoint[1] + eps / 2, size=pointCount)
        ), dtype=self.type).T
        spbDev = cl.Buffer(self.ctx,
                           flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=spbHost)

        x_f_vec = numpy.empty((pointCount, iterations), dtype=self.type)
        x_f_vec_buffer = cl.Buffer(
            context=self.ctx,
            flags=cl.mem_flags.WRITE_ONLY,
            size=x_f_vec.nbytes
        )
        y_f_vec = numpy.empty((pointCount, iterations), dtype=self.type)
        y_f_vec_buffer = cl.Buffer(
            context=self.ctx,
            flags=cl.mem_flags.WRITE_ONLY,
            size=y_f_vec.nbytes
        )
        x_b_vec = numpy.empty((pointCount, iterations), dtype=self.type)
        x_b_vec_buffer = cl.Buffer(
            context=self.ctx,
            flags=cl.mem_flags.WRITE_ONLY,
            size=x_b_vec.nbytes
        )
        y_b_vec = numpy.empty((pointCount, iterations), dtype=self.type)
        y_b_vec_buffer = cl.Buffer(
            context=self.ctx,
            flags=cl.mem_flags.WRITE_ONLY,
            size=y_b_vec.nbytes
        )
        
        self.prg.homoclinic_plot(
            self.queue, (,), None,
            self.type(a),
            self.type(b),
            numpy.int32(iterations),
            spfDev,
            spbDev,
            x_f_vec_buffer,
            y_f_vec_buffer,
            x_b_vec_buffer,
            y_b_vec_buffer
        )

        cl.enqueue_copy(self.queue, x_f_vec, x_f_vec_buffer)
        cl.enqueue_copy(self.queue, y_f_vec, y_f_vec_buffer)
        cl.enqueue_copy(self.queue, x_b_vec, x_b_vec_buffer)
        cl.enqueue_copy(self.queue, y_b_vec, y_b_vec_buffer)

        plot_f.clear()
        plot_b.clear()
        for idx in range(1, pointCount):
            plot_f.scatter(x_f_vec[idx*iterations: (idx+1)*iterations],
                           y_f_vec[idx*iterations: (idx+1)*iterations], s=.5)
            plot_b.scatter(x_b_vec[idx*iterations: (idx+1)*iterations],
                           y_b_vec[idx*iterations: (idx+1)*iterations], s=.5)

class Test(SimpleApp):

    def __init__(self):
        super().__init__("231")

        self.qwe = Homoclinic(self.ctx, self.queue, LOSI)

        self.qwe(
            (0, 0), 1e-3,
            pointCount=10, iterations=10000, a=1.4, b=0.3,

        )