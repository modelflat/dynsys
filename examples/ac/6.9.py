from common import *
from dynsys import SimpleApp, vStack, createSlider
from collections import defaultdict
from time import perf_counter


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
            diffs[i].x = (d_.v.x - r_.v.x);
            diffs[i].y = (d_.v.y - r_.v.y);
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
        assert len(drives) == len(responses) == len(params)
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

    def compute_one(self, queue, skip, iter, drv, rsp, eps):
        drives = numpy.array([drv], dtype=self.system_t)
        responses = numpy.array([rsp], dtype=self.system_t)
        params = numpy.array([(eps,)], dtype=self.param_t)

        res = self._call_sample(queue, skip, iter, drives, responses, params)

        return res

    def compute_eps_range(self, queue, skip, iter, drv, rsp, eps):
        n = len(eps)
        drives = numpy.array([drv] * n, dtype=self.system_t)
        responses = numpy.array([rsp] * n, dtype=self.system_t)
        params = numpy.array([(e,) for e in eps], dtype=self.param_t)

        res = self._call_sample(queue, skip, iter, drives, responses, params)

        return res


def detect_phases(arr: numpy.ndarray, eps):
    thr = arr.max() * eps
    # thr = eps
    return (abs(arr) > thr).astype(numpy.int32), thr


def compute_distr(arr):
    d = defaultdict(lambda: 0)
    c = 0
    for e in arr:
        if e == 1 and c != 0:
            d[c] += 1
            c = 0
        elif e == 0:
            c += 1

    distr_arr = numpy.empty(shape=(len(d), 2))
    for idx, i in enumerate(d.items()):
        distr_arr[idx] = i

    return distr_arr


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("6.9")
        self.im = IntermittencyDetector(self.ctx)

        self.figure = Figure(figsize=(16, 16))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(2, 3)

        self.figure.tight_layout(pad=2.0)

        self.eps_slider, eps_slider = createSlider("r", (0, 1), withLabel="eps = {}", labelPosition="top")

        layout = vStack(
            eps_slider,
            self.canvas,
        )
        self.setLayout(layout)

        self.skip = 0
        self.iter = 1 << 10
        self.eps_c = 0.365
        self.phase_detection = 0.15

        self.eps_slider.valueChanged.connect(self.compute_and_draw)

        self.compute_and_draw()
        self.compute_mean_distr_plot()

    def compute_and_draw(self, *_):
        eps = self.eps_slider.value()
        r, = self.im.compute_one(
            self.queue,
            skip=self.skip,
            iter=self.iter,
            drv=((0.1, 0.1), 1.4, 0.30),
            rsp=((0.1, 0.1), 1.4, 0.28),
            eps=eps
        )

        x_phases, x_thr = detect_phases(r["x"], self.phase_detection)
        y_phases, y_thr = detect_phases(r["y"], self.phase_detection)

        x_distr = compute_distr(x_phases)
        y_distr = compute_distr(y_phases)

        self.ax[0, 1].clear()
        self.ax[0, 1].plot(range(self.iter), r["x"])
        self.ax[0, 1].axhline(x_thr, color="black", linestyle="--")
        self.ax[0, 1].axhline(-x_thr, color="black", linestyle="--")

        self.ax[1, 1].clear()
        self.ax[1, 1].plot(range(self.iter), r["y"])
        self.ax[1, 1].axhline(y_thr, color="black", linestyle="--")
        self.ax[1, 1].axhline(-y_thr, color="black", linestyle="--")

        self.ax[0, 2].clear()
        self.ax[0, 2].hist(x_distr)

        self.ax[1, 2].clear()
        self.ax[1, 2].hist(y_distr)

        self.ax[0, 0].clear()
        self.ax[0, 0].scatter(r["x"], r["y"], s=1.0)

        # self.ax[1, 0].scatter(self.eps_c - eps, a1, s=1.0, color="r")

        self.canvas.draw()

    def compute_mean_distr_plot(self):
        eps = numpy.arange(0, 1, 0.05)

        t = perf_counter()
        res = self.im.compute_eps_range(
            self.queue,
            skip=self.skip,
            iter=self.iter,
            drv=((0.1, 0.1), 1.4, 0.30),
            rsp=((0.1, 0.1), 1.4, 0.28),
            eps=eps
        )
        print("points computed in {:.3f} s".format(perf_counter() - t))

        x = list()
        y = list()
        d = list()

        t = perf_counter()
        for r, e in zip(res, eps):
            x_phases, x_thr = detect_phases(r["x"], self.phase_detection)
            y_phases, y_thr = detect_phases(r["y"], self.phase_detection)

            x_distr = compute_distr(x_phases)
            y_distr = compute_distr(y_phases)

            d.append(self.eps_c - e)
            x.append(x_distr[:, 0].sum() / x_distr[:, 1].sum())
            y.append(y_distr[:, 0].sum() / y_distr[:, 1].sum())

        self.ax[1, 0].clear()
        self.ax[1, 0].plot(d, x)
        print("drawn in {:.3f} s".format(perf_counter() - t))

        # self.ax[1, 1].clear()


if __name__ == '__main__':
    App().run()