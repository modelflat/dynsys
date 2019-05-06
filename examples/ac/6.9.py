from common import *
from dynsys import SimpleApp, vStack, createSlider


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
    const global system_t* d,
    const global system_t* r,
    const global param_t* p,
    
    global val_t* diffs
) {
    const int id = get_global_id(0);
    
    d += id;
    r += id;
    p += id;
    diffs += id * iter;
    
    system_t d_ = *d, r_ = *r;
    param_t p_ = *p;
    
    for (int i = 0; i < skip; ++i) {
        do_step(&d_, &r_, &p_);
    }
    
    for (int i = 0; i < iter; ++i) {
        {
            diffs[i].x = abs(d_.v.x - r_.v.x);
            diffs[i].y = abs(d_.v.y - r_.v.y);
        }
        do_step(&d_, &r_, &p_);
    }
}

"""


class IntermittencyDetector:

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

    def _call_sample(self, queue, skip, iter, drives, responses, params):
        """
        kernel void sample(
            const int skip,
            const int iter,
            const global system_t* d,
            const global system_t* r,
            const global param_t* p,

            global val_t* diffs
        )
        """
        d = copy_dev(self.ctx, drives)
        r = copy_dev(self.ctx, responses)
        p = copy_dev(self.ctx, params)

        diff = numpy.empty((len(drives), iter), dtype=self.val_t)
        diff_dev = alloc_like(self.ctx, diff)

        self.prg.sample(
            queue, (len(drives),), None,
            numpy.int32(skip), numpy.int32(iter),
            d, r, p, diff_dev
        )

        cl.enqueue_copy(queue, diff, diff_dev)

        return diff

    def compute_one(self, queue, iter, drive, response, eps):
        drives = numpy.array([drive], dtype=self.system_t)
        responses = numpy.array([response], dtype=self.system_t)
        params = numpy.array([(eps,)], dtype=self.param_t)

        res = self._call_sample(queue, 0, iter, drives, responses, params)

        return res


def phases(diff):
    


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("6.9")
        self.im = IntermittencyDetector(self.ctx)

        self.figure = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(2, 1)
        self.figure.tight_layout()

        self.eps_slider1, eps_slider1 = createSlider("r", (0, 1), withLabel="eps = {}", labelPosition="top")
        self.eps_slider2, eps_slider2 = createSlider("r", (0, 1), withLabel="eps = {}", labelPosition="top")

        layout = vStack(
            eps_slider1,
            self.canvas,
            eps_slider2,
        )
        self.setLayout(layout)

        self.eps_slider1.valueChanged.connect(self.draw1)
        self.eps_slider2.valueChanged.connect(self.draw2)

        self.iter = 8192

        self.draw1()
        self.draw2()

    def compute(self, eps):
        return self.im.compute_one(
            self.queue,
            iter=self.iter,
            drive=   ((0.1, 0.1), 1.4, 0.3),
            response=((0.1, 0.1), 1.4, 0.31),
            eps=eps
        )

    def draw1(self, *_):
        res, = self.compute(self.eps_slider1.value())

        self.ax[0].clear()
        self.ax[0].plot(range(self.iter), res["x"])
        self.canvas.draw()

    def draw2(self, *_):
        res, = self.compute(self.eps_slider2.value())

        self.ax[1].clear()
        self.ax[1].plot(range(self.iter), res["x"])
        self.canvas.draw()




if __name__ == '__main__':
    App().run()