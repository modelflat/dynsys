from common import *
from dynsys import allocateImage, ParameterizedImageWidget, SimpleApp, createSlider, hStack, vStack
from dynsys.ui.ImageWidgets import *


SOURCE = r"""
// Henon map -- hard-coded
inline void do_step(system_t* d, system_t* r, param_t* p) {
    val_t d_, r_;
    
    #define _F(v, a) (1 - a*v.x*v.x + v.y)
    
    d_.x = _F(d->v, d->a);
    d_.y = d->b * d->v.x;
    
    r_.x = _F(r->v, r->a) + p->eps * (_F(d->v, r->a) - _F(r->v, r->a));
    r_.y = r->b * r->v.x;
    
    d->v = d_; r->v = r_;
}

kernel void sample(
    const int skip,
    const int iter,
    
    const global system_t* d_,
    const global system_t* r_,
    const global param_t*  p_,
    
    global val_t* out_d,
    global val_t* out_r
) {
    const int id = get_global_id(0);
    
    system_t d = d_[id], r = r_[id];
    param_t p = p_[id];
    
    for (int i = 0; i < skip; ++i) {
        do_step(&d, &r, &p);
    }
    
    out_d += id * iter;
    out_r += id * iter;
    
    for (int i = 0; i < iter; ++i) {
        { // output
            out_d[i] = d.v;
            out_r[i] = r.v;
        }
        do_step(&d, &r, &p);
    }
}

#define real double
#define real2 double2

kernel compute_drive_nn(
    const int span,
    const global real2* d, // This should not be real2 but val_t instead
    global real* s
) {
    d += get_global_id(1) * get_global_size(0);
    s += get_global_id(1) * get_global_size(0);
    const int i = get_global_id(0);
    const int left = max(i - span, 0);
    const int right = min(i + span, get_global_size(0) - 1);
    
    real2 cur = d[i];
    real ml = length(d[left] - cur);
    for (int j = left + 1; j < right; ++j) {
        if (j == i) continue;
        const real l = length(d[j] - cur);
        if (l < ml) ml = l;
    }
    
    s[i] = ml;
}

"""


space_shape = (-10, -10, 10, 10)


class NearestNeighbour:

    def __init__(self, ctx):
        val_t_src, self.val_t = make_type(
            ctx=ctx,
            type_name="val_t",
            type_desc=[
                ("x", numpy.float64),
                ("y", numpy.float64)
            ]
        )
        system_t_src, self.system_t = make_type(
            ctx=ctx,
            type_name="system_t",
            type_desc=[
                ("v", self.val_t),
                ("a", numpy.float64),
                ("b", numpy.float64),
            ]
        )
        param_t_src, self.param_t = make_type(
            ctx=ctx,
            type_name="param_t",
            type_desc=[
                ("eps", numpy.float64)
            ]
        )

        sources = [
            val_t_src, system_t_src, param_t_src, SOURCE
        ]

        self.prg = cl.Program(ctx, "\n".join(sources)).build()
        self.ctx = ctx

    def _call_compute_time_series(self, queue, skip, iter, drives, responses, params):
        """
        kernel void assist_system(
            const int skip,
            const int iter,

            const global system_t* d_,
            const global system_t* r_,
            const global param_t* p_,

            global val_t* out_d,
            global val_t* out_r,
        )
        """
        assert len(drives) == len(responses) == len(params)
        n = len(drives)

        drives_dev = copy_dev(self.ctx, drives)
        responses_dev = copy_dev(self.ctx, responses)
        params_dev = copy_dev(self.ctx, params)

        out = numpy.empty((2, n, iter), dtype=self.val_t)
        out_d_dev = alloc_like(self.ctx, out[0])
        out_r_dev = alloc_like(self.ctx, out[1])

        self.prg.sample(
            queue, (n,), None,
            numpy.int32(skip),
            numpy.int32(iter),
            drives_dev, responses_dev, params_dev, out_d_dev, out_r_dev
        )

        cl.enqueue_copy(queue, out[0], out_d_dev)
        cl.enqueue_copy(queue, out[1], out_r_dev)

        return out

    def compute_range_of_eps(self, queue, iter, drive, response, eps):
        n = len(eps)

        drives = numpy.array([drive,]*n, dtype=self.system_t)
        responses = numpy.array([response,]*n, dtype=self.system_t)
        params = numpy.empty((n,), dtype=self.param_t)
        params["eps"] = eps

        res = self._call_compute_time_series(queue, 0, iter, drives, responses, params )

        d, r = res[0], res[1]





class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("2.1")
        self.nn = NearestNeighbour(self.ctx)
        self.figure = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(self.figure)

        self.setLayout(
            hStack(
                self.canvas
            )
        )

        self.compute()

    def compute(self, *_):
        t = time.perf_counter()

        ab = (1.4, 0.3) #self.param_sel.value()
        ab_ass = self.param_assist_sel.value()
        init = self.init_sel.value()
        init_ass = (0.1, 0.2) #self.init_assist_sel.value()

        img_XX, img_YY = self.nn.compute(
            self.queue,
            skip=0,
            iter=1 << 14,
            drives=numpy.array(
                [
                    (init, *ab)
                ], dtype=self.nn.system_t
            ),
            responses=numpy.array(
                [
                    (init, *ab)
                ], dtype=self.nn.system_t
            ),
            assistants=numpy.array(
                [
                    (init_ass, *ab_ass)
                ], dtype=self.nn.system_t
            ),
            params=numpy.array(
                [
                    (self.eps_slider.value(),)
                ], dtype=self.nn.param_t
            )
        )

        self.image_XX_wgt.setTexture(img_XX)
        self.image_YY_wgt.setTexture(img_YY)
        t = time.perf_counter() - t

        # print("{:.3f} s compute, {:.3f} s draw".format(t, tt))


if __name__ == '__main__':
    App().run()