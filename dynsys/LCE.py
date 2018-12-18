import numpy
import pyopencl as cl
from dynsys import allocateImage


DEFS = r"""
#define real double
#define real2 double2
#define real3 double3

#define vec_t real3
"""


SOURCE_RK4 = r"""

#define NL 3
#define N (NL + 1)

#define R(v) (real)(v)

// #define __modif __local
#define __modif __private

void fn_System(real, __modif vec_t[N], __modif const real[NP]);
void fn_System(
    real t,
    __modif vec_t y[N],
    __modif const real p[NP]
) {
    (void)t;
    y[1] = userFn_VARIATION(y[0].x, y[0].y, y[0].z, y[1].x, y[1].y, y[1].z, p[0], p[1], p[2]);
    y[2] = userFn_VARIATION(y[0].x, y[0].y, y[0].z, y[2].x, y[2].y, y[2].z, p[0], p[1], p[2]);
    y[3] = userFn_VARIATION(y[0].x, y[0].y, y[0].z, y[3].x, y[3].y, y[3].z, p[0], p[1], p[2]);
    y[0] = userFn_SYSTEM(y[0].x, y[0].y, y[0].z, p[0], p[1], p[2]);
}


void rk4(int, real, real, __modif vec_t[N], __modif const real[NP]);
void rk4(
    int steps, 
    real time, 
    real tLast, 
    __modif vec_t y[N], 
    __modif const real p[NP]
) {
    real h = (tLast - time) / steps;
    __modif real3 k[N];
    __modif real3 r[N];
    
    for (int i = 0; i < steps; ++i) {
        for (int j = 0; j < N; ++j) {
            k[j] = y[j];
        }
        
        // k1
        fn_System(time, k, p);
        for (int j = 0; j < N; ++j) {
            k[j] *= h / R(2.0);
            r[j] = k[j] / R(3.0);      
            k[j] += y[j];
        }
        
        // k2
        fn_System(time + h / R(2.0), k, p);
        for (int j = 0; j < N; ++j) {
            k[j] *= h;
            r[j] += k[j] / R(3.0);
            k[j] = y[j] + k[j] / R(2.0);
        }
        
        // k3
        fn_System(time + h / R(2.0), k, p);
        for (int j = 0; j < N; ++j) {
            k[j] *= h;
            r[j] += k[j] / R(3.0);
            k[j] += y[j];
        }
        
        // k4
        fn_System(time + h, k, p);
        for (int j = 0; j < N; ++j) {
            y[j] += r[j] + h * k[j] / R(6.0);
        }
        
        time += h;
    }   
}
"""


SOURCE_LYAP = r"""
void LCE(
    vec_t y[N], 
    real param[N - 1], 
    real tStart, real tStep, int iter, int stepIter, 
    real L[N - 1]
) {
    real gsc[N - 1];
    real norms[N - 1];
    real S[N - 1];
    real t = tStart;
    
    for (int i = 0; i < N - 1; ++i) {
        S[i] = 0;
    }
    
    for (int i = 0; i < iter; ++i) {
        // Evolve solution for some time
        rk4(stepIter, t, t + tStep, y, param);
        
        // Renormalize according to gram-schmidt
        for (int j = 0; j < N - 1; ++j) {
            for (int k = 0; k < j; ++k) {
                gsc[k] = dot(y[j + 1], y[k + 1]);
            }
            for (int k = 0; k < j; ++k) {
                y[j + 1] -= gsc[k] * y[k + 1];
            }
            norms[j] = length(y[j + 1]);
            y[j + 1] /= norms[j];
        }
        
        // Accumulate sum of log of norms
        for (int j = 0; j < N - 1; ++j) {
            S[j] += log(norms[j]);
        }
        
        t += tStep;
    }
    
    for (int i = 0; i < N - 1; ++i) {
        L[i] = S[i] / (t - tStart);
    }
}
"""


LYAP_KERNEL = r"""

kernel void computeLCESingle(
    const global real* y0,
    const global real* params,
    real tStart, real tStep,
    int iter, int stepIter,
    global real* L
) {
    vec_t y[N];
    for (int i = 0; i < N; ++i) {
        y[i] = vload3(i, y0);
    }
    
    real p[NP];
    for (int i = 0; i < NP; ++i) {
        p[i] = params[i];
    }
    
    real L_[N - 1];
    LCE(y, p, tStart, tStep, iter, stepIter, L_);
    
    for (int i = 0; i < N - 1; ++i) {
        L[i] = L_[i];
    }
}
"""


LYAPSERIES_KERNEL = r"""
kernel void computeLCEParamVarying(
    const global real* y0,
    const global real* params,
    int paramId,
    real paramMin, real paramMax,
    real tStart, real tStep,
    int iter, int stepIter,
    global real* L
) {
    vec_t y[N];
    for (int i = 0; i < N; ++i) {
        y[i] = vload3(i, y0);
    }
    
    real p[NP];
    for (int i = 0; i < NP; ++i) {
        p[i] = params[i];
    }
    
    p[paramId] = paramMin + get_global_id(0) * (paramMax - paramMin) / get_global_size(0);

    real L_[N - 1];
    LCE(y, p, tStart, tStep, iter, stepIter, L_);
    
    L += get_global_id(0) * (N - 1);
    for (int i = 0; i < N - 1; ++i) {
        L[i] = L_[i];
    }
}
"""


LYAPMAP_KERNEL = r"""
kernel void computeLCEMap(
    const global real* y0,
    const global real* params,
    int lyapIndex,
    real fixedParamValue,
    const real2 param1Bounds,
    const real2 param2Bounds,
    
) {
    vec_t y[N];
    for (int i = 0; i < N; ++i) {
        y[i] = vload3(i, y0);
    }
    
    real p[NP];
    for (int i = 0; i < NP; ++i) {
        p[i] = params[i];
    }
    
    p[0] = param1Bounds.s0 + get_global_id(0) * (param1Bounds.s1 - param1Bounds.s0) / get_global_size(0);
    p[1] = param2Bounds.s0 + get_global_id(1) * (param2Bounds.s1 - param2Bounds.s0) / get_global_size(1);
    p[2] = fixedParamValue;

    real L_[N - 1];
    LCE(y, p, tStart, tStep, iter, stepIter, L_);
    
    real v = L_[lyapIndex];
    
    
}

"""


def dummyOption():
    return "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])


class Lyapunov:

    def __init__(self, ctx, queue, fn):
        self.ctx, self.queue = ctx, queue
        self.prg = cl.Program(self.ctx, "\n".join((DEFS, fn, SOURCE_RK4, SOURCE_LYAP, LYAP_KERNEL))).build(
            options=[dummyOption()]
        )
        self.type = numpy.float64

    def __call__(self, y0: tuple, p1, p2, p3, t0, dt=1, t1=None, iter=2000, stepIter=200):
        y = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf=numpy.array((*y0, *numpy.eye(3).flat), dtype=self.type))
        param = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=numpy.array((p1, p2, p3), dtype=self.type))
        lyapHost = numpy.empty((3,), dtype=self.type)
        lyapDev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=lyapHost.nbytes)

        self.prg.computeLCESingle(
            self.queue, (1,), None,
            y, param,
            self.type(t0),
            self.type(dt),
            numpy.int32(iter),
            numpy.int32(stepIter),
            lyapDev
        )

        cl.enqueue_copy(self.queue, lyapHost, lyapDev)

        return lyapHost


class LyapunovSeries:

    def __init__(self, ctx, queue, fn, paramCount):
        self.ctx, self.queue = ctx, queue
        self.prg = cl.Program(self.ctx, "\n".join((DEFS, fn, SOURCE_RK4, SOURCE_LYAP, LYAPSERIES_KERNEL))).build(
            options=[dummyOption()]
        )
        self.type = numpy.float64
        self.paramCount = paramCount

    def __call__(self, y0: tuple, params, paramId, paramLinspace, t0, dt=1, t1=None, iter=2000, stepIter=200):
        y = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf=numpy.array((*y0, *numpy.eye(3).flat), dtype=self.type))

        assert self.paramCount == len(params)

        param = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=numpy.array(params, dtype=self.type))
        lyapHost = numpy.empty((paramLinspace.shape[0], 3), dtype=self.type)
        lyapDev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=lyapHost.nbytes)

        self.prg.computeLCEParamVarying(
            self.queue, (paramLinspace.shape[0],), None,
            y, param,
            numpy.int32(paramId),
            self.type(paramLinspace[0]),
            self.type(paramLinspace[-1]),
            self.type(t0),
            self.type(dt),
            numpy.int32(iter),
            numpy.int32(stepIter),
            lyapDev
        )

        cl.enqueue_copy(self.queue, lyapHost, lyapDev)

        return lyapHost


class LyapunovMap:

    def __init__(self, ctx, queue, fn, imageShape):
        self.ctx, self.queue = ctx, queue
        self.prg = cl.Program(self.ctx, "\n".join((DEFS, fn, SOURCE_RK4, SOURCE_LYAP, LYAPMAP_KERNEL))).build(
            options=[dummyOption()]
        )
        self.type = numpy.float64
        self.imageShape = imageShape
        self.hostImage, self.deviceImage = allocateImage(self.ctx, imageShape)

    def clear(self, readBack=False, color=(1.0, 1.0, 1.0, 1.0)):
        cl.enqueue_fill_image(
            self.queue, self.deviceImage,
            color=numpy.array(color, dtype=numpy.float32),
            origin=(0,)*len(self.imageShape), region=self.imageShape
        )
        if readBack:
            self.readFromDevice()

    def readFromDevice(self):
        cl.enqueue_copy(
            self.queue, self.hostImage, self.deviceImage,
            origin=(0,)*len(self.imageShape), region=self.imageShape
        )
        return self.hostImage

    def __call__(self, y0: tuple, p1, p2, p3, t0, dt=1, t1=None, iter=2000, stepIter=200):
        y = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf=numpy.array((*y0, *numpy.eye(3).flat), dtype=self.type))
        param = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=numpy.array((p1, p2, p3), dtype=self.type))
        lyapHost = numpy.empty((3,), dtype=self.type)
        lyapDev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=lyapHost.nbytes)

        self.prg.LCEmap(
            self.queue, self.imageShape, None,
            numpy.int32(iter),
            numpy.int32(stepIter),
            self.type(t0),
            self.type(dt),
            y, param, lyapDev
        )

        cl.enqueue_copy(self.queue, lyapHost, lyapDev)

        return lyapHost
