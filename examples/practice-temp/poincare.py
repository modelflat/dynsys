from dynsys import SimpleApp

from dynsys.LCE import dummyOption

import time
from typing import Iterable

import numpy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from dynsys import LCE
from dynsys import SimpleApp, vStack, QLabel, Image2D, hStack

import numpy
import pyopencl as cl

DEFS = r"""
#define real double
#define real3 double3
#define vec_t real3
#define R(v) (real)(v)
#define LOAD vload3
#define STORE vstore3
"""

SOURCE_RK4 = r"""

void fn_System(real, vec_t*, const real[NP]);
void fn_System(real t, vec_t* y, const real p[NP]) {
    (void)t;
    (*y) = userFn_SYSTEM((*y).x, (*y).y, (*y).z, p[0], p[1], p[2]);
}

void rk4(int, real, real, vec_t*, const real[NP]);
void rk4(
    int steps, 
    real time, 
    real tLast, 
    vec_t* y, 
    const real p[NP]
) {
    real h = (tLast - time) / steps;
    vec_t k;
    vec_t r;
    
    for (int i = 0; i < steps; ++i) {
        k = *y;
        // printf("0 %.3f %.3f %.3f\n", k.x, k.y, k.z);
        
        // k1
        fn_System(time, &k, p);
        k *= h / R(2.0);
        r = k / R(3.0);      
        k += *y;
        
        //printf("1 %.3f %.3f %.3f\n", k.x, k.y, k.z);
        //printf("0 %.3f %.3f %.3f\n", r.x, r.y, r.z);
        
        // k2
        fn_System(time + h / R(2.0), &k, p);
        k *= h;
        r += k / R(3.0);
        k = *y + k / R(2.0);
        
        //printf("2 %.3f %.3f %.3f\n", k.x, k.y, k.z);
        //printf("1 %.3f %.3f %.3f\n", r.x, r.y, r.z);
        
        // k3
        fn_System(time + h / R(2.0), &k, p);
        k *= h;
        r += k / R(3.0);
        k += *y;
        
        //printf("3 %.3f %.3f %.3f\n", k.x, k.y, k.z);
        //printf("2 %.3f %.3f %.3f\n", r.x, r.y, r.z);
        
        // k4
        fn_System(time + h, &k, p);
        (*y) = *y + r + h * k / R(6.0);
        
        //printf("4 %.3f %.3f %.3f\n", (*y).x, (*y).y, (*y).z);
        //printf("3 %.3f %.3f %.3f\n", r.x, r.y, r.z);
        
        time += h;
    }   
}

"""

SOURCE_POINCARE = r"""
kernel void poincare(
    const global real* y0,
    const global real* params,
    int varId,
    real varValue,
    real tStart,
    real tStep,
    int iter,
    int skip,
    int maxRes,
    global int* count,
    global real* result
) {
    real p[NP];
    for (int i = 0; i < NP; ++i) {
        p[i] = params[i];
    }
   
    vec_t y = LOAD(0, y0);
    vec_t yPrev = y;
    
    real t = tStart;
    int resPos = 0;
    for (int i = 0; i < iter; ++i) {
        //printf("%.3f %.3f %.3f\n", y.x, y.y, y.z);
        rk4(1, t, t + tStep, &y, p);
        
        if (i > skip) {
            if        (varId == 0) {
                if (y.x < varValue && yPrev.x >= varValue && resPos < maxRes) {
                    STORE(y, resPos++, result);
                }
            } else if (varId == 1) {
                if (y.y < varValue && yPrev.y >= varValue && resPos < maxRes) {
                    STORE(y, resPos++, result);
                }
            } else if (varId == 2) {
                if (y.z < varValue && yPrev.z >= varValue && resPos < maxRes) {
                    STORE(y, resPos++, result);
                }
            }
        }
        
        yPrev = y;
    }
    
    *count = resPos;
}
"""

Fn_KPR = r"""
#define userFn_SYSTEM(x, y, z, h, g, eps) (vec_t)( \
    2*h*x + y - g*z, \
    -x, \
    (x - 8.592*z + 22*z*z - 14.408*z*z*z) / eps \
)

#define userFn_VARIATION(x, y, z, x_, y_, z_, h, g, eps) (vec_t)( \
    2*h*x_ + y_ - g*z_, \
    -x_, \
    (x_ - 8.592*z_ + 44*z_*z - 43.224*z_*z*z) / eps \
)

#define NP 3
"""


class Poincare:
    def __init__(self, ctx, queue, fn):
        self.ctx, self.queue = ctx, queue
        self.prg = cl.Program(self.ctx, "\n".join((DEFS, fn, SOURCE_RK4, SOURCE_POINCARE))).build(
            options=[dummyOption()]
        )
        self.type = numpy.float64
        self.paramCount = 3

    def __call__(self, y0: tuple, params: tuple, varId, varValue, t0, dt=1.0, skip=None, iter=2000, maxPoints=32):
        y = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf=numpy.array((*y0, *numpy.eye(3).flat), dtype=self.type))

        assert self.paramCount == len(params)

        param = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=numpy.array(params, dtype=self.type))
        resultHost = numpy.empty((maxPoints, 3), dtype=self.type)
        resultDev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=resultHost.nbytes)
        countHost = numpy.empty((1,), dtype=numpy.int32)
        countDev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=countHost.nbytes)

        if skip is None:
            skip = iter // 4 * 3

        self.prg.poincare(
            self.queue, (1,), None,
            y, param, numpy.int32(varId), self.type(varValue),
            self.type(t0), self.type(dt),
            numpy.int32(iter), numpy.int32(skip), numpy.int32(maxPoints),
            countDev, resultDev
        )

        cl.enqueue_copy(self.queue, countHost, countDev)
        cl.enqueue_copy(self.queue, resultHost, resultDev)

        return resultHost[:countHost[0]]


class Test(SimpleApp):

    def __init__(self):
        super().__init__("123")
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.poincareSimple(self.figure)

        self.setLayout(
            hStack(
                vStack(self.canvas),
            )
        )

    def poincareSimple(self, fig):
        poincare = Poincare(self.ctx, self.queue, Fn_KPR)
        points = poincare((0, 1, 0), (.07, .85, 0.2), 0, .5, t0=0, dt=1e-3, iter=100000)
        sub = fig.add_subplot(111)
        sub.plot(*points.T[1:], "r.")


if __name__ == '__main__':
    Test().run()
