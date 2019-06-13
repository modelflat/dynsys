from matplotlib.ticker import MultipleLocator

from common import *
from dynsys import SimpleApp, vStack, hStack, createSlider


EQUATIONS_PYRAGAS = r"""

// Coupled Henon maps
// - Equations:
//   - Driver:
#define USER_equation_d_x(x1, y1, a1, b1, x2, y2, a2, b2) \
    1 - a1*x1*x1 + y1
#define USER_equation_d_y(x1, y1, a1, b1, x2, y2, a2, b2) \
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

#define f(x, y, a, b) (1 - a*x*x + y)

// Coupled Henon maps
// - Equations:
//   - Driver:
#define USER_equation_d_x(x1, y1, a1, b1, x2, y2, a2, b2) \
    f(x1, y1, a1, b1) + eps*(f(x2, y2, a2, b2) - f(x1, y1, a1, b1))
#define USER_equation_d_y(x1, y1, a1, b1, x2, y2, a2, b2) \
    b1*x1
//   - Response:
#define USER_equation_r_x(x1, y1, a1, b1, x2, y2, a2, b2) \
    f(x2, y2, a2, b2) + eps*(f(x1, y1, a1, b1) - f(x2, y2, a2, b2))
#define USER_equation_r_y(x1, y1, a1, b1, x2, y2, a2, b2) \
    b2*x2

// - Variations:
//   - Driver:
#define USER_variation_d_x(x1, y1, x1_, y1_, a1, b1, x2, y2, x2_, y2_, a2, b2) \
    y1_ - 2*a1*x1*x1_ + eps*(-2*a2*x2*x2_ + y2_ + 2*a1*x1*x1_ - y1_)
#define USER_variation_d_y(x1, y1, x1_, y1_, a1, b1, x2, y2, x2_, y2_, a2, b2) \
    b1*x1_
//   - Response:
#define USER_variation_r_x(x1, y1, x1_, y1_, a1, b1, x2, y2, x2_, y2_, a2, b2) \
    y2_ - 2*a2*x2*x2_ + eps*(-2*a1*x1*x1_ + y1_ + 2*a2*x2*x2_ - y2_)
#define USER_variation_r_y(x1, y1, x1_, y1_, a1, b1, x2, y2, x2_, y2_, a2, b2) \
    b2*x2_

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
            d.x, d.y, v.d.x, v.d.y, d_->a, d_->b,
            r.x, r.y, v.r.x, v.r.y, r_->a, r_->b
        );
        v_[i].d.y = USER_variation_d_y(
            d.x, d.y, v.d.x, v.d.y, d_->a, d_->b,
            r.x, r.y, v.r.x, v.r.y, r_->a, r_->b
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

    d_->v.x = USER_equation_d_x(d.x, d.y, d_->a, d_->b,
                                r.x, r.y, r_->a, r_->b);
    d_->v.y = USER_equation_d_y(d.x, d.y, d_->a, d_->b,
                                r.x, r.y, r_->a, r_->b);
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

    def compute_range_of_eps(self, queue, t_start, t_step, iter, drives, responses, eps_range):
        params = numpy.empty((len(drives),), dtype=self.param_t)
        params["eps"] = eps_range
        variations = self._make_variations(len(drives))
        return self._call_compute_cle(queue, t_start, t_step, iter, drives, responses, variations, params)

    def compute_range_of_eps_same_systems(self, queue, iter, drv, res, eps):
        n = len(eps)
        drives = numpy.array(n*[drv,], dtype=self.system_t)
        responses = numpy.array(n*[res,], dtype=self.system_t)
        variations = self._make_variations(n)
        params = numpy.empty(n, dtype=self.param_t)
        params["eps"] = eps
        return self._call_compute_cle(queue, 0, 1, iter, drives, responses, variations, params)


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("CLE")
        self.cle = CLE(self.ctx)

        self.figure = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(self.figure)

        self.iter_slider, it_sl_el = createSlider("i", (1, 8192),
                                                  withLabel="iter = {}",labelPosition="top")
        self.response_b_slider, rb_sl_el = createSlider("r", (.2, .4), withLabel="b = {}",
                                                        labelPosition="top")

        self.iter_slider.valueChanged.connect(self.compute_and_draw)
        self.response_b_slider.valueChanged.connect(self.compute_and_draw)

        self.setLayout(vStack(
            # it_sl_el,
            # rb_sl_el,
            self.canvas
        ))

        self.compute_and_draw()

    def compute_cle_series(self):
        eps = numpy.array([*numpy.linspace(0.09, 0.1075, 500),
                           # *numpy.linspace(0.08, 0.2, 250)
                           ], dtype=numpy.float64)
        return eps, self.cle.compute_range_of_eps_same_systems(
            self.queue,
            iter=1 << 16,
            # iter=self.iter_slider.value(),
            drv=((0.0, 0.0), 1.4, 0.3),
            res=((0.1, 0.1), 1.4, 0.3),
            eps=eps
        )

    def compute_and_draw(self, *_):
        eps, lyap = self.compute_cle_series()

        self.figure.clear()
        ax = self.figure.subplots(1, 1)

        print(lyap)

        for i in range(4):
            ax.plot(eps[1:], lyap.T[i][1:], label=f"L{i}")

        # lp = lyap.T[2][0]
        # for i,l in enumerate(lyap.T[2][1:]):
        #     if l < 0 and lp > 0:
        #         print(eps[i])
        #     lp = l

        spacing = 0.00025
        minorLocator = MultipleLocator(spacing)
        # Set minor tick locations.
        ax.xaxis.set_minor_locator(minorLocator)
        # ax.yaxis.set_minor_locator(minorLocator)

        # ax.plot(eps, lyap.T[2], label="L0 of response")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xlabel("ε")
        ax.set_ylabel("L")
        ax.grid(which="both")
        ax.legend()

        # self.figure.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    App().run()
