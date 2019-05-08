from common import *
from dynsys import SimpleApp, vStack, hStack, createSlider
from tqdm import tqdm

EQUATIONS_PYRAGAS = r"""

// Coupled Henon maps
// - Equations:
//   - Driver:
#define USER_equation_d_x(x1, y1, a1, b1) \
    1 - a1*x1*x1 + y1
#define USER_equation_d_y(x1, y1, a1, b1) \
    b1*x1 + D*random(&(p->rng_state))
//   - Response:
#define USER_equation_r_x(x1, y1, a1, b1, x2, y2, a2, b2) \
    1 - a2*x2*x2 + y2 + eps*(-a1*x1*x1 + y1 + a2*x2*x2 - y2)
#define USER_equation_r_y(x1, y1, a1, b1, x2, y2, a2, b2) \
    b2*x2 + D*random(&(p->rng_state))

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
    x1 + D*random(&(p->rng_state))
//   - Response:
#define USER_equation_r_x(x1, y1, a1, b1, x2, y2, a2, b2) \
    a2 - (eps*x1*x2 + (1 - eps)*x2*x2) + b2*y2
#define USER_equation_r_y(x1, y1, a1, b1, x2, y2, a2, b2) \
    x2 + D*random(&(p->rng_state))

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
    double D = p->D;
    uint2* rng_state = &(p->rng_state);
    
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


inline void do_step_simple(system_t* d_, system_t* r_, param_t* p) {
    double eps = p->eps;
    double D = p->D;
    
    val_t d = d_->v, r = r_->v;

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


kernel void sample(
    const int skip,
    const int iter,
    const global system_t* d,
    const global system_t* r,
    const global param_t* p,
    
    global val_t* d_out,
    global val_t* r_out
) {
    const int id = get_global_id(0);
    
    d += id;
    r += id;
    p += id;
    d_out += id * iter;
    r_out += id * iter;
    
    system_t d_ = *d, r_ = *r;
    param_t p_ = *p;
    
    for (int i = 0; i < skip; ++i) {
        do_step_simple(&d_, &r_, &p_);
    }
    
    for (int i = 0; i < iter; ++i) {
        {
            d_out[i] = d_.v;
            r_out[i] = r_.v;
        }
        do_step_simple(&d_, &r_, &p_);
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
                ("D", numpy.float64),
                ("rng_state", cl.cltypes.uint2),
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
            PRNG_SOURCE,
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

    def _call_sample(self, queue, skip, iter, d, r, p):
        """
        kernel void sample(
            const int skip,
            const int iter,
            const global system_t* d,
            const global system_t* r,
            const global param_t* p,

            global val_t* d_out,
            global val_t* r_out
        )
        """
        drive = copy_dev(self.ctx, d)
        response = copy_dev(self.ctx, r)
        param = copy_dev(self.ctx, p)

        out = numpy.empty((2, iter), dtype=self.val_t)
        d_out_dev = alloc_like(self.ctx, out[0])
        r_out_dev = alloc_like(self.ctx, out[1])

        self.prg.sample(
            queue, (1,), None,
            numpy.int32(skip), numpy.int32(iter),
            drive, response, param, d_out_dev, r_out_dev
        )

        cl.enqueue_copy(queue, out[0], d_out_dev)
        cl.enqueue_copy(queue, out[1], r_out_dev)

        return out

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

    def _make_seed(self, i=None):
        if i is not None:
            return 42 * i, 24 * i
        return numpy.random.randint(1 << 31), numpy.random.randint(1 << 31)

    def compute_range_of_eps(self, queue, t_start, t_step, iter, drives, responses, eps_range, D):
        params = numpy.empty((len(drives),), dtype=self.param_t)
        params["eps"] = eps_range
        params["D"] = D
        params["rng_state"] = [self._make_seed(1) for i in range(len(drives))]
        variations = self._make_variations(len(drives))
        return self._call_compute_cle(queue, t_start, t_step, iter, drives, responses, variations, params)

    def compute_range_of_eps_same_systems(self, queue, iter, drv, res, eps, D):
        n = len(eps)
        drives = numpy.array(n*[drv,], dtype=self.system_t)
        responses = numpy.array(n*[res,], dtype=self.system_t)
        variations = self._make_variations(n)
        params = numpy.empty(n, dtype=self.param_t)
        params["eps"] = eps
        params["D"] = D
        params["rng_state"] = [self._make_seed(1) for i in range(n)]
        return self._call_compute_cle(queue, 0, 1, iter, drives, responses, variations, params)

    def compute_phase_plot(self, queue, iter, drv, res, eps, D):
        drives = numpy.array([drv,], dtype=self.system_t)
        responses = numpy.array([res,], dtype=self.system_t)
        params = numpy.empty((1,), dtype=self.param_t)
        params["eps"] = eps
        params["D"] = D
        params["rng_state"] = [self._make_seed(1)]
        return self._call_sample(queue, 0, iter, drives, responses, params)


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("7.2 - CLE")
        self.cle = CLE(self.ctx)

        self.figure = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(1, 2)
        self.figure.tight_layout(pad=1.5)

        self.response_b_slider, rb_sl_el = createSlider(
            "r", (.2, .3), withLabel="b = {}", labelPosition="top", withValue=0.25
        )
        self.d_slider, d_sl_el = createSlider(
            "r", (0., 0.1), withLabel="D = {}", labelPosition="top"
        )
        self.eps_slider, eps_sl_el = createSlider(
            "r", (0., 1.0), withLabel="eps = {}", labelPosition="top"
        )

        self.response_b_slider.valueChanged.connect(self.compute_and_draw)
        self.d_slider.valueChanged.connect(self.compute_and_draw)
        self.eps_slider.valueChanged.connect(self.compute_and_draw)

        self.setLayout(vStack(
            eps_sl_el,
            d_sl_el,
            rb_sl_el,
            self.canvas
        ))

        self.iter = 1 << 14

        self.compute_and_draw()

    def compute_cle_series(self):
        eps = numpy.linspace(0, 1.0, 500)
        return eps, self.cle.compute_range_of_eps_same_systems(
            self.queue,
            iter=self.iter,
            drv=((0.1, 0.1), 1.4, 0.3),
            res=((0.1, 0.1), 1.4, 0.25), #self.response_b_slider.value()),
            eps=eps,
            D=self.d_slider.value()
        )

    def compute_phase(self):
        return self.cle.compute_phase_plot(
            self.queue,
            iter=self.iter,
            # iter=self.iter_slider.value(),
            drv=((0.1, 0.1), 1.4, 0.3),
            res=((0.1, 0.1), 1.4, 0.25),# self.response_b_slider.value()),
            eps=self.eps_slider.value(),
            D=0.01#self.d_slider.value()
        )

    def compute_and_draw(self, *_):
        eps, lyap = self.compute_cle_series()

        lp = lyap.T[2][0]
        for i,l in enumerate(lyap.T[2][1:]):
            if lp > 0 > l:
                print(eps[i])
            lp = l

        self.ax[0].clear()
        self.ax[0].plot(eps, lyap.T[2], label="L0 of response")
        self.ax[0].axhline(0, color="black", linestyle="--")
        self.ax[0].axvline(self.eps_slider.value(), color="red", linestyle="--")
        self.ax[0].set_xlabel("ε")
        self.ax[0].set_ylabel("L")
        self.ax[0].legend()

        d, r = self.compute_phase()

        self.ax[1].clear()
        self.ax[1].scatter(d["x"], d["y"], s=0.5)

        # self.ax[2].clear()
        # self.ax[2].scatter(r["x"], r["y"], s=0.5)

        self.canvas.draw()


class D_EPS_App(SimpleApp):

    def __init__(self):
        super(D_EPS_App, self).__init__("7.2")
        self.cle = CLE(self.ctx)

        self.figure = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(1, 1)
        self.figure.tight_layout(pad=2.5)

        self.response_b_slider, rb_sl_el = createSlider(
            "r", (.1, .3), withLabel="b = {}", labelPosition="top", withValue=0.25
        )

        # self.response_b_slider.valueChanged.connect(self.compute_and_draw)

        self.setLayout(vStack(
            rb_sl_el,
            self.canvas
        ))

        self.compute_and_draw()

    def compute_e_c(self, D, b):
        eps = numpy.linspace(0, 1.0, 500)
        lyap = self.cle.compute_range_of_eps_same_systems(
            self.queue,
            iter=1 << 14,
            drv=((0.1, 0.1), 1.4, 0.3),
            res=((0.1, 0.1), 1.4, b),
            eps=eps, D=D
        )
        L = lyap.T[2]
        return eps[numpy.nanargmax(L < 0)]

        # return numpy.nan

    def compute_and_draw(self, *_):
        d_range = numpy.linspace(0.0, 0.0095, 200)
        b_range = numpy.linspace(0.25, 0.35, 50)

        D = 0.01
        b = 0.2
        self.response_b_slider.setValue(b)

        points = []
        for D in tqdm(d_range):
        # for b in tqdm(b_range):
            points.append((D, self.compute_e_c(D, b)))

        self.ax.clear()
        self.ax.plot(*numpy.array(points).T)
        self.ax.set_xlabel("D")
        self.ax.set_ylabel("ε critical")

        self.canvas.draw()


if __name__ == '__main__':
    d_eps = False

    if d_eps:
        D_EPS_App().run()
    else:
        App().run()
