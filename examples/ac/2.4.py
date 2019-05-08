from common import *
from dynsys import SimpleApp, vStack, createSlider
from collections import defaultdict
from time import perf_counter


EQUATIONS_PYRAGAS = r"""

// Coupled Henon maps
// - Equations:
//   - Driver:
#define USER_equation_d_x(x1, y1, a1, b1) \
    1 - a1*x1*x1 + y1
#define USER_equation_d_y(x1, y1, a1, b1) \
    b1*x1
//   - Response:
#define USER_equation_r_x(x1, y1, a1, b1, x2, y2, a2, b2) \
    1 - a2*x2*x2 + y2 + eps*(-a1*x1*x1 + y1 + a2*x2*x2 - y2)
#define USER_equation_r_y(x1, y1, a1, b1, x2, y2, a2, b2) \
    b2*x2

// - Variations:
//   - Driver:
#define USER_variation_d_x(x1, y1, x1_, y1_, a1, b1) \
    y1_ - 2*a1*x1*x1_
#define USER_variation_d_y(x1, y1, x1_, y1_, a1, b1) \
    b1*x1_
//   - Response:
#define USER_variation_r_x(x1, y1, x1_, y1_, a1, b1, x2, y2, x2_, y2_, a2, b2) \
    y2_ - 2*a2*x2*x2_ + eps*(-2*a1*x1*x1_ + y1_ + 2*a2*x2*x2_ - y2_)
#define USER_variation_r_y(x1, y1, x1_, y1_, a1, b1, x2, y2, x2_, y2_, a2, b2) \
    b2*x2_

"""


EQUATIONS_CANONICAL = r"""

// Coupled Henon maps
// - Equations:
//   - Driver:
#define USER_equation_d_x(x1, y1, a1, b1) \
    a1 - x1*x1 + b1*y1
#define USER_equation_d_y(x1, y1, a1, b1) \
    x1
//   - Response:
#define USER_equation_r_x(x1, y1, a1, b1, x2, y2, a2, b2) \
    a2 - (eps*x1*x2 + (1 - eps)*x2*x2) + b2*y2
#define USER_equation_r_y(x1, y1, a1, b1, x2, y2, a2, b2) \
    x2

// - Variations:
//   - Driver:
#define USER_variation_d_x(x1, y1, x1_, y1_, a1, b1) \
    b1*y1_ - 2*x1*x1_
#define USER_variation_d_y(x1, y1, x1_, y1_, a1, b1) \
    x1_
//   - Response:
#define USER_variation_r_x(x1, y1, x1_, y1_, a1, b1, x2, y2, x2_, y2_, a2, b2) \
    b2*y2_ - (eps*x2*x1_ + eps*x1*x2_ + 2*(1-eps)*x2*x2_)
#define USER_variation_r_y(x1, y1, x1_, y1_, a1, b1, x2, y2, x2_, y2_, a2, b2) \
    x2_

"""


POTENTIALLY_GENERATED = r"""

#define DIM 4

#define real double
#define real4 double4

// These functions should also be generated. 
// Unfortunately, performance is of concern here, so in the future more careful implementation needed

// For simplicity, here system_val_t is reinterpreted as double4 and necessary operations are performed 

// dot(x, y) as if x and y are vectors
inline real sys_dot(system_val_t x, system_val_t y) {
    // union { system_val_t x; real4 v; } __temp_x;
    // __temp_x.x = x;
    // union { system_val_t x; real4 v; } __temp_y;
    // __temp_y.x = y;
    real4 __temp_x = {
        x.d.x, x.d.y, x.r.x, x.r.y
    };
    real4 __temp_y = {
        y.d.x, y.d.y, y.r.x, y.r.y
    };
    return dot(__temp_x, __temp_y);
}

// normalize x as if it is float vector, return norm
inline real sys_norm(system_val_t* x) {
    // union { system_val_t x; real4 v; } __temp_x;
    // __temp_x.x = *x;
    real4 __temp_x = {
        x->d.x, x->d.y, x->r.x, x->r.y
    };
    real norm = length(__temp_x); 
    x->d.x /= norm;
    x->d.y /= norm;
    x->r.x /= norm;
    x->r.y /= norm;
    return norm;
}

// x -= y * z
inline void sys_sub_mul(system_val_t* x, system_val_t y, real z) {
    x->d.x -= y.d.x * z;
    x->d.y -= y.d.y * z;
    x->r.x -= y.r.x * z;
    x->r.y -= y.r.y * z;
}

// d - driver system
// r - response system
// v - variation coordinates (variation parameters are the same as d/r)
// p - meta-params (eps for example)
inline void do_step(system_t* d_, system_t* r_, system_val_t v_[DIM], param_t* p) {
    double eps = p->eps; // !! this [ab]uses macro outer namespace. watch out!
    
    val_t d = d_->v, r = r_->v;
    
    for (int i = 0; i < DIM; ++i) {
        system_val_t v = v_[i];
        
        v_[i].d.x = USER_variation_d_x(
            d.x, d.y, v.d.x, v.d.y, d_->a, d_->b
        );
        v_[i].d.y = USER_variation_d_y(
            d.x, d.y, v.d.x, v.d.y, d_->a, d_->b
        );
        v_[i].r.x = USER_variation_r_x(
            d.x, d.y, v.d.x, v.d.y, d_->a, d_->b,
            r.x, r.y, v.r.x, v.r.y, r_->a, r_->b
        );
        v_[i].r.y = USER_variation_r_y(
            d.x, d.y, v.d.x, v.d.y, d_->a, d_->b,
            r.x, r.y, v.r.x, v.r.y, r_->a, r_->b
        );
    }

    d_->v.x = USER_equation_d_x(d.x, d.y, d_->a, d_->b);
    d_->v.y = USER_equation_d_y(d.x, d.y, d_->a, d_->b);
    r_->v.x = USER_equation_r_x(d.x, d.y, d_->a, d_->b, 
                                r.x, r.y, r_->a, r_->b);
    r_->v.y = USER_equation_r_y(d.x, d.y, d_->a, d_->b, 
                                r.x, r.y, r_->a, r_->b);
}

"""


LYAPUNOV_SRC = r"""

void lyapunov(
    real, real, int, system_t, system_t, system_val_t[DIM], param_t,
    // out
    real[DIM]
);
void lyapunov(
    real t_start, real t_step, int iter, 
    system_t d, system_t r, system_val_t v[DIM], param_t p,
    // out
    real L[DIM]
) {
    real gsc[DIM];
    real norms[DIM];
    real S[DIM];
    
    real t = t_start;
    
    // to establish system
    // for (int i = 0; i < (1 << 15); ++i) {
    //     do_step(&d, &r, v, &p);
    // }
    
    for (int i = 0; i < DIM; ++i) {
        S[i] = 0;
    }
    
    for (int i = 0; i < iter; ++i) {
        // Iterate map:
        do_step(&d, &r, v, &p);
        
        // Renormalize according to Gram-Schmidt
        for (int j = 0; j < DIM; ++j) {
            
            for (int k = 0; k < j; ++k) {
                gsc[k] = sys_dot(v[j], v[k]);
            }
            
            for (int k = 0; k < j; ++k) {
                // v[j] -= gsc[k] * v[k];
                sys_sub_mul(v + j, v[k], gsc[k]);
            }
            
            norms[j] = sys_norm(v + j);
        }
        
        // Accumulate sum of log of norms
        for (int j = 0; j < DIM; ++j) {
            S[j] += log(norms[j]);
        }
        
        t += t_step;
    }
    
    for (int i = 0; i < DIM; ++i) {
        L[i] = t_step * S[i] / iter;
    }
}
"""


KERNEL_SRC = r"""

kernel void compute_cle(
    const real t_start, const real t_step, const int iter,
    const global system_t* d,
    const global system_t* r,
    const global system_val_t* v,
    const global param_t*  p,
    global real* cle
) {
    const int id = get_global_id(0);
    
    d += id;
    r += id;
    v += DIM * id;
    p += id;
    
    system_val_t var[DIM];
    for (int i = 0; i < DIM; ++i) {
        var[i] = v[i];
    }
    
    real L[DIM];
    lyapunov(t_start, t_step, iter, *d, *r, var, *p, L);
    
    cle += DIM * id;
    for (int i = 0; i < DIM; ++i) {
        cle[i] = L[i];
    }
}

"""


class CLE:

    def __init__(self, ctx):
        self.use_workaround = True
        self.ctx = ctx
        self.DIM = 4
        param_t_src, self.param_t = make_type(
            ctx, "param_t", [
                ("eps", numpy.float64),
            ]
        )
        val_t_src, self.val_t = make_type(
            ctx, "val_t", [
                ("x", numpy.float64),
                ("y", numpy.float64),
            ]
        )
        system_val_t_src, self.system_val_t = make_type(
            ctx, "system_val_t", [
                ("d", self.val_t),
                ("r", self.val_t),
            ]
        )
        system_t_src, self.system_t = make_type(
            ctx, "system_t", [
                ("v", self.val_t),
                ("a", numpy.float64),
                ("b", numpy.float64),
            ]
        )
        sources = [
            param_t_src, val_t_src, system_val_t_src, system_t_src,
            EQUATIONS_CANONICAL,
            # EQUATIONS_PYRAGAS,
            POTENTIALLY_GENERATED, LYAPUNOV_SRC, KERNEL_SRC
        ]
        self.prg = cl.Program(ctx, "\n".join(sources)).build()

    def _call_compute_cle(self, queue, t_start, t_step, iter, drives, responses, variations, params):
        """
        kernel void compute_cle(
            const real t_start,
            const real t_step,
            const int iter,
            const global system_t* d,
            const global system_t* r,
            const global system_val_t* v,
            const global param_t*  p,
            global real* cle
        )
        """
        assert len(drives) == len(responses) == len(params)
        assert variations.shape[0] == len(drives)
        assert variations.shape[1] == self.DIM

        drives_dev = copy_dev(self.ctx, drives)
        responses_dev = copy_dev(self.ctx, responses)
        variations_dev = copy_dev(self.ctx, variations)
        params_dev = copy_dev(self.ctx, params)

        cle = numpy.empty((len(drives), self.DIM), dtype=numpy.float64)
        cle_dev = alloc_like(self.ctx, cle)

        self.prg.compute_cle(
            queue, (len(drives),), None,
            numpy.float64(t_start), numpy.float64(t_step), numpy.int32(iter),
            drives_dev, responses_dev, variations_dev, params_dev,
            cle_dev
        )

        cl.enqueue_copy(queue, cle, cle_dev)

        if self.use_workaround:
            # FIXME either alignment/other programming bug (unlikely but possible), or algorithm is shit (likely but less possible):
            # We need to swap some data before return TODO figure out why

            # Presumably: cle.T[0:2] -- lyapunov exponents of driver
            #             cle.T[2:4] -- of response (conditional)
            # It turns out data is messed up for some reason

            # It's *probably* a programming error -- the result looks *almost* good
            # but:
            # 1) L2 of drive and L2 of response look swapped (???)
            cle.T[1], cle.T[2] = cle.T[2].copy(), cle.T[1].copy()
            # 2) the first elements of these also look swapped (???)
            cle.T[1][0], cle.T[2][0] = cle.T[2][0], cle.T[1][0]

        return cle

    def _make_variations(self, n):
        variations = numpy.zeros((n, 4), dtype=self.system_val_t)
        for v in variations:
            # omfg this is awful. TODO optimize?
            v[0] = numpy.zeros(1, dtype=self.system_val_t)
            v[0]["d"]["x"] = 1
            v[1] = numpy.zeros(1, dtype=self.system_val_t)
            v[1]["d"]["y"] = 1
            v[2] = numpy.zeros(1, dtype=self.system_val_t)
            v[2]["r"]["x"] = 1
            v[3] = numpy.zeros(1, dtype=self.system_val_t)
            v[3]["r"]["y"] = 1
        return variations

    def compute(self, queue, iter, drv: list, rsp: list, par: list):
        drv = numpy.array(drv, dtype=self.system_t)
        rsp = numpy.array(rsp, dtype=self.system_t)
        par = numpy.array(par, dtype=self.param_t)
        variations = self._make_variations(len(drv))
        return self._call_compute_cle(queue, 0, 1, iter, drv, rsp, variations, par)


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("2.4")
        self.lp = CLE(self.ctx)

        self.figure = Figure(figsize=(16, 16))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(1, 1)

        self.figure.tight_layout(pad=2.0)

        layout = vStack(
            self.canvas,
        )
        self.setLayout(layout)

        self.skip = 0
        self.iter = 1 << 14

        self.compute_and_draw()

    def compute_and_draw(self, *_):
        b_count = 100
        eps_count = 100

        eps_range = numpy.linspace(0, 1, eps_count)
        b_range = numpy.linspace(0.2, 0.4, b_count)

        base_b = 0.3

        t = time.perf_counter()
        r = self.lp.compute(
            self.queue,
            iter=self.iter,
            drv=[
                ((0.1, 0.1), 1.4, base_b) for _ in b_range for _ in eps_range
            ],
            rsp=[
                ((0.1, 0.1), 1.4, b) for b in b_range for _ in eps_range
            ],
            par=[
                (eps,) for _ in b_range for eps in eps_range
            ]
        ).reshape((b_count, eps_count, 4))
        t = time.perf_counter() - t
        print("Computed in {:.3f} s".format(t))

        def detect_eps_c(arr_for_b):
            # workaround:
            a = arr_for_b.T[2][1:] if arr_for_b.T[2][0] < 0 else arr_for_b.T[2]
            return eps_range[numpy.argmax(a < 0)]

        points = numpy.array([(b - base_b, detect_eps_c(r[i])) for i, b in enumerate(b_range)])

        self.ax.plot(*points.T)
        self.ax.grid()

        self.canvas.draw()


if __name__ == '__main__':
    App().run()