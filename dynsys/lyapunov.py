import numpy
import pyopencl as cl
import pyopencl.cltypes
from PyQt5.QtWidgets import QApplication, QDesktopWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import os
import time
from PyQt5.Qt import QVBoxLayout, QHBoxLayout, QLayout, QColor, QWidget, QLabel
from PyQt5.QtCore import Qt, pyqtSignal as Signal
import sys


from core import create_context_and_queue

def vStack(*args, cm=(0, 0, 0, 0)):
    l = QVBoxLayout()
    l.setContentsMargins(*cm)
    l.setSpacing(2)
    for a in args:
        if isinstance(a, QLayout):
            l.addLayout(a)
        else:
            l.addWidget(a)
    return l





os.environ["PYOPENCL_NO_CACHE"] = "1"


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


def copy_dev(ctx, buf):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=buf)


def alloc_like(ctx, buf):
    return cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=buf.nbytes)



PRNG_SOURCE = r"""
// source code:
// http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
uint MWC64X(uint2 *state);
uint MWC64X(uint2 *state) {
    enum { A = 4294883355U };
    uint x = (*state).x, c = (*state).y;  // Unpack the state
    uint res = x^c;                     // Calculate the result
    uint hi = mul_hi(x,A);              // Step the RNG
    x = x*A+c;
    c = hi+(x<c);
    *state = (uint2)(x,c);               // Pack the state back up
    return res;                        // Return the next result
}

void init_state(ulong seed, uint2* state);
void init_state(ulong seed, uint2* state) {
    int id = get_global_id(0) + 1;
    uint2 s = as_uint2(seed);
    (*state) = (uint2)(
        // create a mixture of id and two seeds
        (id + s.x & 0xFFFF) * s.y,
        (id ^ (s.y & 0xFFFF0000)) ^ s.x
    );
}

// retrieve random float in range [0.0; 1.0] (both inclusive)
inline float random(uint2* state) {
    return ((float)MWC64X(state)) / (float)0xffffffff;
}
"""

# from dynsys import SimpleApp, vStack, hStack, createSlider


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
    y1_ - 2*a1*x1*x1_+ eps*(-2*a2*x2*x2_ + y2_ + 2*a1*x1*x1_ - y1_)
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

${system.common_typedefs()}

<%
    system_t = system.get_type_name("system")
    system_var_t = system.get_type_name("variable")
    param_t = system.get_type_name("parameter")
    dim = system.get_dim()
%>

typedef ${system.get_type_definition("system")} ${system_t};
typedef ${system.get_type_definition("variable")} ${system_var_t};
typedef ${system.get_type_definition("parameter")} ${param_t};

${system.request_capability("variation_step", "lyap_step")}
${system.request_capability("variable_dot_product", "lyap_dot")}
${system.request_capability("variable_sub_mul", "lyap_sub_mul")}
${system.request_capability("variable_euc_norm", "lyap_norm")}

private void lyapunov(
// input
    real            t_start, 
    real            t_step, 
    int             iter, 
    ${system_var_t} s,
    ${system_var_t} v[${dim}], 
    ${param_t}      p,
// output
    real L[${dim}]
) {
    real gsc  [${dim}];
    real norms[${dim}];
    real S    [${dim}];
    real t = t_start;
    
    for (int i = 0; i < ${dim}; ++i) {
        S[i] = 0;
    }
    
    for (int i = 0; i < iter; ++i) {
        lyap_step(t_step, &s, &p, v);
        
        // Renormalize according to Gram-Schmidt
        for (int j = 0; j < ${dim}; ++j) {
            
            for (int k = 0; k < j; ++k) {
                gsc[k] = lyap_dot(v[j], v[k]);
            }
            
            for (int k = 0; k < j; ++k) {
                // v[j] -= gsc[k] * v[k];
                lyap_sub_mul(v + j, v[k], gsc[k]);
            }
            
            norms[j] = lyap_norm(v + j);
        }
        
        // Accumulate sum of log of norms
        for (int j = 0; j < ${dim}; ++j) {
            S[j] += log(norms[j]);
        }
        
        t += t_step;
    }
    
    for (int i = 0; i < ${dim}; ++i) {
        L[i] = t_step * S[i] / iter;
    }
}
"""


KERNEL_SRC = r"""

% for mode in ["SSMP", "MSSP"]:
kernel void compute_cle_${mode}(
    const real t_start, 
    const real t_step, 
    const int iter,
    const global ${system_var_t}* s,
    const global ${param_t}*      p,
    const global ${system_var_t}* v,
    // output
    global real* cle
) {
    const int id = get_global_id(0);
    
% if mode == "MSSP":
    s += id;
% endif
% if mode == "SSMP":
    p += id;
% endif
    
    v += ${dim} * id;
    
    ${system_var_t} var[${dim}];
    for (int i = 0; i < ${dim}; ++i) {
        var[i] = v[i];
    }
    
    real L[${dim}];
    lyapunov(t_start, t_step, iter, *s, *p, var, L);
    
    cle += ${dim} * id;
    for (int i = 0; i < ${dim}; ++i) {
        cle[i] = L[i];
    }
}
% endfor
"""


class CLE:

    def __init__(self, ctx, system):
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
        from mako.template import Template
        sources = Template("\n".join([
            # param_t_src, val_t_src, system_val_t_src, system_t_src,
            # EQUATIONS_CANONICAL,
            # EQUATIONS_PYRAGAS,
            # POTENTIALLY_GENERATED,
            LYAPUNOV_SRC, KERNEL_SRC
        ])).render(system=system)

        print(sources)
        self.prg = cl.Program(ctx, sources).build()

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

        from system import System

        self.cle = CLE(self.ctx, System(
            [], [], [
                "123", "321", "890", "098"
            ], capabilities={
                "variation_step": """   
// s - system
// p - meta-params (eps for example)
// v - variation coordinates
inline void ${name}(
    ${system.get_type_name("variable")}*   s_,
    ${system.get_type_name("parameter")}* p, 
    ${system.get_type_name("variable")}    v_[DIM]
) {
    real eps = p->eps; // !! this [ab]uses macro outer namespace. watch out!
    
    for (int i = 0; i < ${system.get_dim()}; ++i) {
        system_val_t v = v_[i];
    
    ${system.generate("variations")}
    
    ${system.generate("equations")}
    
//         ${syste} = USER_variation_d_x(
//             d.x, d.y, v.d.x, v.d.y, d_->a, d_->b,
//             r.x, r.y, v.r.x, v.r.y, r_->a, r_->b
//         );
//         v_[i].d.y = USER_variation_d_y(
//             d.x, d.y, v.d.x, v.d.y, d_->a, d_->b,
//             r.x, r.y, v.r.x, v.r.y, r_->a, r_->b
//         );
//         v_[i].r.x = USER_variation_r_x(
//             d.x, d.y, v.d.x, v.d.y, d_->a, d_->b,
//             r.x, r.y, v.r.x, v.r.y, r_->a, r_->b
//         );
//         v_[i].r.y = USER_variation_r_y(
//             d.x, d.y, v.d.x, v.d.y, d_->a, d_->b,
//             r.x, r.y, v.r.x, v.r.y, r_->a, r_->b
//         );
//   }
//    d_->v.x = USER_equation_d_x(d.x, d.y, d_->a, d_->b,
//                                r.x, r.y, r_->a, r_->b);
//    d_->v.y = USER_equation_d_y(d.x, d.y, d_->a, d_->b,
//                                r.x, r.y, r_->a, r_->b);
//    r_->v.x = USER_equation_r_x(d.x, d.y, d_->a, d_->b, 
//                                r.x, r.y, r_->a, r_->b);
//    r_->v.y = USER_equation_r_y(d.x, d.y, d_->a, d_->b, 
//                                r.x, r.y, r_->a, r_->b);

}
                """,
                "variable_dot_product": """
// dot(x, y) as if x and y are vectors
inline real ${name}(${system.get_type_name("variable")} x, ${system.get_type_name("variable")} y) {
    return dot(x, y);
}
                """,
                "variable_sub_mul": """
inline void ${name}(${system.get_type_name("variable")}* x, ${system.get_type_name("variable")}* y, real z) {
    *x -= y * z;
}
                """,
                "variable_euc_norm": """
// normalize x as if it is float vector, return norm
inline real ${name}(${system.get_type_name("variable")}* x) {
    real norm = length(__temp_x); 
    *x /= norm;
    return norm;
}
                """
            }
        ))

        self.figure = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(self.figure)

        # self.iter_slider, it_sl_el = createSlider("i", (1, 8192),
        #                                           withLabel="iter = {}",labelPosition="top")
        # self.response_b_slider, rb_sl_el = createSlider("r", (.2, .4), withLabel="b = {}",
        #                                                 labelPosition="top")

        # self.iter_slider.valueChanged.connect(self.compute_and_draw)
        # self.response_b_slider.valueChanged.connect(self.compute_and_draw)

        self.setLayout(vStack(
            # it_sl_el,
            # rb_sl_el,
            self.canvas
        ))

        self.compute_and_draw()

    def compute_cle_series(self):
        eps = numpy.linspace(0, 1.0, 100)
        return eps, self.cle.compute_range_of_eps_same_systems(
            self.queue,
            iter=1 << 12,
            # iter=self.iter_slider.value(),
            drv=((1e-4, -1e-4), 1.4, 0.3),
            res=((1e-4, 1e-4), 1.4, 0.3),
            eps=eps
        )

    def compute_and_draw(self, *_):
        eps, lyap = self.compute_cle_series()

        self.figure.clear()
        ax = self.figure.subplots(1, 1)

        print(lyap)

        for i in range(4):
            ax.plot(eps, lyap.T[i], label="L{} of {}".format(
                i % 2, "drive" if i < 2 else "resp."
            ))

        # lp = lyap.T[2][0]
        # for i,l in enumerate(lyap.T[2][1:]):
        #     if l < 0 and lp > 0:
        #         print(eps[i])
        #     lp = l


        # ax.plot(eps, lyap.T[2], label="L0 of response")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xlabel("ε")
        ax.set_ylabel("L")
        ax.legend()

        # self.figure.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    App().run()
