import numpy
import pyopencl as cl
from dynsys import SimpleApp, allocateImage

Fn = r"""
#define userFn_SYSTEM(x, y, z, sig, bet, r) (real3)( \
    (sig)*((y) - (x)), \
    -(x)*(z) + (r)*(x) - (y), \
    (x)*(y) - (bet)*(z)\
)

#define userFn_VARIATION(x, y, z, x_, y_, z_, sig, bet, r) (real3)( \
    ((y_) - (x_)) * (sig), \
    ((r) - (z))*(x_) - (y_) - (x)*(z_), \
    (y)*(x_) + (x)*(y_) - (bet)*(z_) \
)
"""


DEFS = r"""
#define real double
#define real3 double3
"""


SOURCE_RK4 = r"""

#define N 4
#define NP 3

#define R(v) (real)(v)

// #define MOD __local
#define MOD __private

void fn_System(real t, MOD real3 y[4], MOD const real p[3]);
void fn_System(real t, MOD real3 y[4], MOD const real p[3]) {
    (void)t;
    y[1] = userFn_VARIATION(y[0].x, y[0].y, y[0].z, y[1].x, y[1].y, y[1].z, p[0], p[1], p[2]);
    y[2] = userFn_VARIATION(y[0].x, y[0].y, y[0].z, y[2].x, y[2].y, y[2].z, p[0], p[1], p[2]);
    y[3] = userFn_VARIATION(y[0].x, y[0].y, y[0].z, y[3].x, y[3].y, y[3].z, p[0], p[1], p[2]);
    y[0] = userFn_SYSTEM(y[0].x, y[0].y, y[0].z, p[0], p[1], p[2]);
}


void rk4(int, real, real, MOD real3[N], MOD const real[NP]);
void rk4(int steps, real time, real tLast, MOD real3 y[N], MOD const real p[NP]) {
    real h = (tLast - time) / steps;
    MOD real3 k[N];
    MOD real3 r[N];
    
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


LYAPUNOV_KERNEL = r"""

void LCE(
    real3 y[N], 
    real3 param[N - 1], 
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

kernel void computeLCESingle(
    const global real* y0,
    const global real* params,
    real tStart, real tStep,
    int iter, int stepIter,
    global real* L
) {
    real3 y[N];
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

kernel void computeLCEParamVarying(
    const global real* y0,
    const global real* params,
    int paramId,
    real paramMin, real paramMax,
    real tStart, real tStep,
    int iter, int stepIter,
    global real* L
) {
    real3 y[N];
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


def dummyOption():
    return "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])


class Lyapunov:

    def __init__(self, ctx, queue, fn):
        self.ctx, self.queue = ctx, queue
        self.prg = cl.Program(self.ctx, "\n".join((DEFS, fn, SOURCE_RK4, LYAPUNOV_KERNEL))).build(
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

    def __init__(self, ctx, queue, fn):
        self.ctx, self.queue = ctx, queue
        self.prg = cl.Program(self.ctx, "\n".join((DEFS, fn, SOURCE_RK4, LYAPUNOV_KERNEL))).build(
            options=[dummyOption()]
        )
        self.type = numpy.float64

    def __call__(self, y0: tuple, p1, p2, p3, paramId, paramBounds, paramTicks, t0, dt=1, t1=None, iter=2000, stepIter=200):
        y = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf=numpy.array((*y0, *numpy.eye(3).flat), dtype=self.type))
        param = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=numpy.array((p1, p2, p3), dtype=self.type))
        lyapHost = numpy.empty((3,), dtype=self.type)
        lyapDev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=lyapHost.nbytes)

        self.prg.computeLCEParamVarying(
            self.queue, (paramTicks,), None,
            y, param,
            numpy.int32(paramId),
            self.type(paramBounds[0]),
            self.type(paramBounds[1]),
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
        self.prg = cl.Program(self.ctx, "\n".join((DEFS, fn, SOURCE_RK4, LYAPUNOV_KERNEL))).build(
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



class Test(SimpleApp):

    def __init__(self):
        super().__init__("123")

        self.lyap = LyapunovSeries(self.ctx, self.queue, Fn)
        print(self.lyap((1, 2, 3), 10, 8/3, 28, 0, ))

        exit()


def rk4(t0, t1, y0, F, steps=100):
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    for i in range(steps):
        # k1
        k = h * F(t, y) / 2.0
        r = k / 3.0

        # k2
        k = h * F(t + h / 2.0, y + k) / 2.0
        r += k * (2.0 / 3.0)

        # k3
        k = h * F(t + h / 2.0, y + k)
        r += k / 3.0

        # k4
        pass

        y += r + h * F(t + h, y + k) / 6.0
        t += h
    return y


def Lorenz(t, X, sig=10, bet=8/3, r=28):
    x, y, z = X[0], X[1], X[2]
    res = numpy.empty(shape=(12,), dtype=numpy.float)

    res[0] = sig*(y - x)

    res[1] = -x*z + r*x - y

    res[2] = x*y - bet*z

    #        m
    # cij = sum aik*bkj
    #       k-1

    # 0 0
    res[3] = -sig * X[3] + sig * X[4]
    # 1 0
    res[4] = (r - z) * X[3] - X[4] - x * X[5]
    # 2 0
    res[5] = y * X[3] + x * X[4] - bet * X[5]
    # 0 1
    res[6] = -sig * X[6] + sig * X[7]
    # 1 1
    res[7] = (r - z) * X[6] - X[7] - x * X[8]
    # 2 1
    res[8] = y * X[6] + x * X[7] - bet * X[8]
    # 0 2
    res[9] = -sig * X[9] + sig * X[10]
    # 1 2
    res[10] = (r - z) * X[9] - X[10] - x * X[11]
    # 2 2
    res[11] = y * X[9] + x * X[10] - bet * X[11]

    return res


def lyapunov(n, fn, tStart, tStep, tEnd, y0):
    totalIter = int((tEnd - tStart) / tStep)

    y = numpy.zeros((n + 1, n), numpy.float)
    S = numpy.zeros((n,), numpy.float)

    gsc = numpy.empty((n,), numpy.float)
    norms = numpy.empty((n,), numpy.float)
    series = numpy.empty((totalIter, n + 1), dtype=numpy.float)

    y[0] = y0
    y[1:] = numpy.eye(n)

    t = tStart

    for i in range(totalIter):
        y = rk4(t, t + tStep, y.flat, fn).reshape((n + 1, n))
        t += tStep

        w = y[1:]
        for j in range(n):
            for k in range(j):
                gsc[k] = numpy.dot(w[j], w[k])

            for k in range(j):
                w[j] -= numpy.dot(gsc[k], w[k])

            norms[j] = numpy.linalg.norm(w[j])
            w[j] /= norms[j]
        y[1:] = w

        S += numpy.log(norms)
        series[i] = (t, *(S / (t - tStart)))

        if (i + 1) % 10 == 0:
            print(*map(lambda x: "%.3f" % x, series[i]))
            print("%.3f" % sum(series[i][1:]))

    return series


if __name__ == '__main__':
    pass#q = lyapunov(3, Lorenz, 0, 1, 300, [1, 2, 3])[-1][1:]

Test().run()
