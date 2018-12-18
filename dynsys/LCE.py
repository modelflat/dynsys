import numpy
import pyopencl as cl
from dynsys import allocateImage


DEFS = r"""
#define real double
#define real2 double2
#define real3 double3
"""


SOURCE_RK4 = r"""

#if (NP == 3)
#define NL 3
#define vec_t real3
#define LOAD vload3
#else
#define NL 2
#define vec_t real2
#define LOAD vload2
#endif

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
    for (int i = 1; i < N; ++i) {
#if (NP == 3) 
        y[i] = userFn_VARIATION(y[0].x, y[0].y, y[0].z, y[i].x, y[i].y, y[i].z, p[0], p[1], p[2]);
#else
        y[i] = userFn_VARIATION(y[0].x, y[0].y, y[i].x, y[i].y, p[0], p[1]);
#endif
    }
#if (NP == 3)
    y[0] = userFn_SYSTEM(y[0].x, y[0].y, y[0].z, p[0], p[1], p[2]);
#else
    y[0] = userFn_SYSTEM(y[0].x, y[0].y, p[0], p[1]);
#endif
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
    __modif vec_t k[N];
    __modif vec_t r[N];
    
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
);
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
        y[i] = LOAD(i, y0);
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

float3 hsv2rgb(float3);
float3 hsv2rgb(float3 hsv) {
    const float c = hsv.y * hsv.z;
    const float x = c * (1 - fabs(fmod( hsv.x / 60, 2 ) - 1));
    float3 rgb;
    if      (0 <= hsv.x && hsv.x < 60) {
        rgb = (float3)(c, x, 0);
    } else if (60 <= hsv.x && hsv.x < 120) {
        rgb = (float3)(x, c, 0);
    } else if (120 <= hsv.x && hsv.x < 180) {
        rgb = (float3)(0, c, x);
    } else if (180 <= hsv.x && hsv.x < 240) {
        rgb = (float3)(0, x, c);
    } else if (240 <= hsv.x && hsv.x < 300) {
        rgb = (float3)(x, 0, c);
    } else {
        rgb = (float3)(c, 0, x);
    }
    return (rgb + (hsv.z - c));
}

kernel void computeLCEMap(
    const global real* y0,
    int lyapIndex,
    const real2 param1Bounds,
    const real2 param2Bounds,
    real param3Value,
    real tStart, 
    real tStep, 
    int iter, int stepIter,
    global real* result
) {
    vec_t y[N];
    for (int i = 0; i < N; ++i) {
        y[i] = vload3(i, y0);
    }
    
    real p[NP];
    p[0] = param1Bounds.s0 + get_global_id(0) * (param1Bounds.s1 - param1Bounds.s0) / get_global_size(0);
    p[1] = param2Bounds.s0 + get_global_id(1) * (param2Bounds.s1 - param2Bounds.s0) / get_global_size(1);
    p[2] = param3Value;

    real L_[N - 1];
    LCE(y, p, tStart, tStep, iter, stepIter, L_);
    result += get_global_id(1) * get_global_size(0) + get_global_id(0);
    *result = L_[lyapIndex];
}

kernel void colorMap(
    const global real* result,
    real min_, real max_,
    write_only image2d_t output
) {
    result += get_global_id(1) * get_global_size(0) + get_global_id(0);
    real v = (*result - min_) / (max_ - min_);
    float4 color = (float4)(hsv2rgb((float3)(v * 240, 1.0, 1.0)), 1.0);
    write_imagef(output, (int2)(get_global_id(0), get_global_id(1)), color);
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

    def __call__(self, y0: tuple, lyapIndex: int,
                 param1Bounds: tuple, param2Bounds: tuple, param3Value: float,
                 t0=0, dt=1, t1=None, iter=2000, stepIter=200):
        y = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf=numpy.array((*y0, *numpy.eye(3).flat), dtype=self.type))

        resultHost = numpy.empty(self.imageShape, dtype=self.type)
        resultDev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=resultHost.nbytes)

        self.prg.computeLCEMap(
            self.queue, self.imageShape, None,
            y,
            numpy.int32(lyapIndex),
            numpy.array(param1Bounds, dtype=self.type),
            numpy.array(param2Bounds, dtype=self.type),
            self.type(param3Value),
            self.type(t0),
            self.type(dt),
            numpy.int32(iter),
            numpy.int32(stepIter),
            resultDev
        )

        cl.enqueue_copy(self.queue, resultHost, resultDev)
        min_, max_ = min(resultHost.flat), max(resultHost.flat)
        print(min_, max_)

        self.prg.colorMap(
            self.queue, self.imageShape, None,
            resultDev, min_, max_, self.deviceImage
        )

        return self.readFromDevice(), min_, max_
