from dynsys import SimpleApp, allocateImage, Image2D, vStack
import pyopencl as cl
import numpy
from matplotlib import pyplot
from dynsys.LCE import dummyOption

DEFS = r"""
#define real double
#define real3 double3
#define vec_t real3
#define NP 3
"""

Fn_Rossler = r"""

#define userFn_SYSTEM(x, y, z, a, b, c) (vec_t)( \
    -y - z, \
    x + a*y, \
    b + z * (x - c) \
)

"""


Fn_KPR = r"""



"""


SOURCE_BIFTREE = r"""

vec_t fn_System(real t, vec_t y, const real p[NP]);
vec_t fn_System(real t, vec_t y, const real p[NP]) {
    (void)t;
    return userFn_SYSTEM(y.x, y.y, y.z, p[0], p[1], p[2]);
}

vec_t rk4(real t, real t1, vec_t y, int steps, const real param[NP]);
vec_t rk4(real t, real t1, vec_t y, int steps, const real param[NP]) {
    vec_t k;
    vec_t r;
    const real h = (t1 - t) / steps;
    for (int i = 0; i < steps; ++i) {
        // 1
        k = h * fn_System(t, y, param) / 2.0;
        r = k / 3.0;
        // 2
        k = h * fn_System(t + h / 2.0, y + k, param) / 2.0;
        r += k * (2.0/3.0);
        // 3
        k = h * fn_System(t + h / 2.0, y + k, param);
        r += k / 3.0;
        //4
        y += r + h * fn_System(t + h, y + k, param) / 6.0;
        t += h;
    }
    return y;
}

int poincare(
    real t, real t1, 
    vec_t y,
    const real param[NP], 
    int varId, real varSlice, 
    int skip, int iter,
    global real* points
) {
    const real h = (t1 - t) / (real)(skip + iter);
    const real tSkip = t + h * skip;
    // skip points:
    y = rk4(t, tSkip, y, skip, param);
    //
    t = tSkip;
    vec_t yPrev = y;
    int count = 0;
    for (int i = 0; i < iter; ++i) {
        y = rk4(t, t + h, y, 1, param);
        t += h;
        
        // detect if trajectory is piercing required plane
        if (varId == 0) {
            if (y.x < varSlice && yPrev.x >= varSlice) {
                vstore3(y, count++, points);
            }
        } else if (varId == 1) {
            if (y.y < varSlice && yPrev.y >= varSlice) {
                vstore3(y, count++, points);
            }
        } else {
            printf("error!\n");
            return count;
        }
        yPrev = y;
    }
    return count;
}

kernel void computePoincareSingle(
    const real3 y0,
    const global real* params,
    real t0, real t1,
    real xSlice,
    int skip,
    int iter,
    global int* counts,
    global real* points
) {
    real par[NP];
    for (int i = 0; i < NP; ++i) {
        par[i] = params[i];
    }
    counts[0] = poincare(t0, t1, y0, par, 0, xSlice, skip, iter, points);
}

int poincareTree(
    real t, real t1, 
    vec_t y,
    const real param[NP], 
    int varId, real varSlice, 
    int skip, int iter,
    int xCoord, real xMin, real xMax,
    write_only image2d_t output
) {
    const real h = (t1 - t) / (real)(skip + iter);
    const real tSkip = t + h * skip;
    // skip points:
    y = rk4(t, tSkip, y, skip, param);
    //
    t = tSkip;
    vec_t yPrev = y;
    int count = 0;
    for (int i = 0; i < iter; ++i) {
        y = rk4(t, t + h, y, 1, param);
        t += h;
        
        // detect if trajectory is piercing required plane
        if (varId == 0) {
            if (y.x < varSlice && yPrev.x >= varSlice) {
                real v = y.y;
                if (v >= xMin && v <= xMax) {
                    real coord = (v - xMin) / (xMax - xMin) * (get_image_height(output));
                    write_imagef(output, 
                        (int2)(xCoord, get_image_height(output) - (int)coord),
                        (float4)(0, 0, 0, 1.0)
                    );
                }
            }
        } else {
            printf("error!\n");
            return count;
        }
        yPrev = y;
    }
    return count;
}

kernel void drawBifurcationTree(
    const real3 y0,
    const global real* params,
    int paramIdx, real paramMin, real paramMax,
    real t0, real t1,
    real xSlice,
    int skip, int iter,
    real xMin, real xMax,
    write_only image2d_t result
) {
    real par[NP];
    for (int i = 0; i < NP; ++i) {
        par[i] = params[i];
    }
    par[paramIdx] = paramMin + (paramMax - paramMin) * (real)get_global_id(0) / (real)get_global_size(0);
    poincareTree(t0, t1, y0, par, 0, xSlice, skip, iter, get_global_id(0), xMin, xMax, result);
}

"""


class Poincare:

    def __init__(self, ctx, queue, fn):
        self.ctx, self.queue = ctx, queue
        self.prg = cl.Program(self.ctx, "\n".join(
            (DEFS, fn, SOURCE_BIFTREE)
        )).build(options=[dummyOption()])
        self.type = numpy.float64

    def __call__(self, startPoint, params, t0, t1, xSlice, skip, iter):
        params = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=numpy.array(params, dtype=self.type))
        countsHost = numpy.empty((1,), dtype=numpy.int32)
        countsDev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=countsHost.nbytes)
        resultHost = numpy.empty((iter, 3), dtype=self.type)
        resultDev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=resultHost.nbytes)

        self.prg.computePoincareSingle(
            self.queue, (1,), None,
            numpy.array(startPoint, dtype=self.type),
            params,
            self.type(t0), self.type(t1),
            self.type(xSlice),
            numpy.int32(skip), numpy.int32(iter),
            countsDev,
            resultDev
        )

        cl.enqueue_copy(self.queue, countsHost, countsDev)
        cl.enqueue_copy(self.queue, resultHost, resultDev)

        print(countsHost[0])

        return resultHost[:countsHost[0]]


class PoincareBifTree:

    def __init__(self, ctx, queue, fn, dim):
        self.ctx, self.queue = ctx, queue
        self.prg = cl.Program(self.ctx, "\n".join(
            (DEFS, fn, SOURCE_BIFTREE)
        )).build(options=[dummyOption()])
        self.type = numpy.float64
        self.imageShape = dim
        self.hostImage, self.deviceImage = allocateImage(self.ctx, dim)

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

    def __call__(self, startPoint, params, t0, t1, xSlice,
                 paramIdx, paramMin, paramMax,
                 xMin, xMax,
                 skip, iter):

        self.clear()

        params = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=numpy.array(params, dtype=self.type))

        self.prg.drawBifurcationTree(
            self.queue, (self.imageShape[0],), None,
            numpy.array(startPoint, dtype=self.type),
            params,
            numpy.int32(paramIdx),
            self.type(paramMin), self.type(paramMax),
            self.type(t0), self.type(t1),
            self.type(xSlice),
            numpy.int32(skip), numpy.int32(iter),
            self.type(xMin), self.type(xMax),
            self.deviceImage
        )

        return self.readFromDevice()



class Test(SimpleApp):

    def __init__(self):
        super().__init__("123")
        self.computeTree()

    def computePoincare(self):
        p = Poincare(self.ctx, self.queue, Fn_Rossler)
        res = p((0, 1, 0), (0.1, 0.1, 14), 0, 1000, xSlice=0.1, skip=100000, iter=4096)
        pyplot.plot(res.T[1], res.T[2], "r.")
        pyplot.show()

    def computeTree(self):
        p = PoincareBifTree(self.ctx, self.queue, Fn_Rossler, (768, 512))
        res = p(
            startPoint=(0, 1, 0),
            params=(0.2, 0.1, 5.7),
            t0=0, t1=1000,
            xSlice=0.0,
            paramIdx=1, paramMin=0, paramMax=2,
            xMin=0, xMax=10,
            skip=50000, iter=100000
        )
        self.label = Image2D()
        self.label.setTexture(res)
        self.setLayout(vStack(self.label))
        return res



if __name__ == '__main__':
    Test().run()
