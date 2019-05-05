from common import *
from dynsys import SimpleApp, createSlider


SERIES_SOURCE = r"""

#define _F(x, y, a) (1 - a*x*x + y)

// HENON MAP hardcoded
inline void do_step(system_t* d_, system_t* r_, param_t* p) {
    system_t d = *d_, r = *r_;
    
    d_->x = _F(d.x, d.y, d.a);
    d_->y = d.b * d.x + p->D * random(&(p->rng_state));
    
    r_->x = _F(r.x, r.y, r.a) + p->eps * (_F(d.x, d.y, d.a) - _F(r.x, r.y, r.a));
    r_->y = r.b * r.x + p->D * random(&(p->rng_state));
}

kernel void compute_time_series(
    global system_t* d_systems,
    global system_t* r_systems,
    global param_t* params,
    
    int skip, 
    int iter,
    
    global double* drive,
    global double* response
) {
    const int id = get_global_id(0);
    system_t d = d_systems[id];
    system_t r = r_systems[id];
    param_t p = params[id];
    
    for (int i = 0; i < skip; ++i) {
        do_step(&d, &r, &p);
    }
    
    drive += 2 * id * iter;
    //       ^ dim
    response += 2 * id * iter;
    //          ^ dim
    
    for (int i = 0; i < iter; ++i) {
        do_step(&d, &r, &p);
        // save drive
        drive[2*i + 0] = d.x;
        drive[2*i + 1] = d.y;
        // save response
        response[2*i + 0] = r.x;
        response[2*i + 1] = r.y;
    }
}

"""


class SystemWithNoise:

    def __init__(self, ctx):
        self.ctx = ctx

        system_t_src, self.system_t = make_type(ctx, "system_t", [
            ("x", numpy.float64),
            ("y", numpy.float64),
            ("a", numpy.float64),
            ("b", numpy.float64),
        ])
        param_t_src, self.param_t = make_type(ctx, "param_t", [
            ("eps", numpy.float64),
            ("D", numpy.float64),
            ("rng_state", cl.cltypes.uint2),
        ])
        sources = [
            PRNG_SOURCE, system_t_src, param_t_src, SERIES_SOURCE
        ]
        self.prg = cl.Program(ctx, "\n".join(sources)).build()

    def compute(self, queue, skip: int, iter: int, eps: float, D: float,
                drive: dict, response: dict):
        num_par_systems = 1  # should be locked to 1 for now

        drive_sys = numpy.empty(num_par_systems, dtype=self.system_t)
        response_sys = numpy.empty(num_par_systems, dtype=self.system_t)

        for i in range(num_par_systems):
            for k, v in drive.items():
                drive_sys[i][k] = v

            for k, v in response.items():
                response_sys[i][k] = v

        drive_sys_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=drive_sys)
        response_sys_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=response_sys)

        params = numpy.empty(num_par_systems, dtype=self.param_t)
        params["eps"] = eps
        params["D"] = D
        params["rng_state"] = (42, 24)

        params_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=params)

        drive = numpy.empty((iter, 2), dtype=numpy.float64)
        drive_dev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=drive.nbytes)

        response = numpy.empty((iter, 2), dtype=numpy.float64)
        response_dev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=response.nbytes)

        self.prg.compute_time_series(
            queue, (num_par_systems,), None,
            drive_sys_dev,
            response_sys_dev,
            params_dev,
            numpy.int32(skip), numpy.int32(iter),
            drive_dev,
            response_dev
        )

        cl.enqueue_copy(queue, drive, drive_dev)
        cl.enqueue_copy(queue, response, response_dev)

        return drive, response


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("6.9")

        self.sys = SystemWithNoise(self.ctx)

        self.figure = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(self.figure)

        self.d_slider, d_sl_elem = createSlider("r", (0, 0.02),
                                                labelPosition="top",
                                                withLabel="D = {}")
        self.d_slider.valueChanged.connect(self.compute_and_draw_phase)
        self.eps_slider, eps_sl_elem = createSlider("r", (0, 1),
                                                    labelPosition="top",
                                                    withLabel="eps = {}")

        self.iter_slider, iter_sl_elem = createSlider("i", (1, 1 << 14))
        self.iter_slider.valueChanged.connect(self.compute_and_draw_phase)

        self.eps_slider.valueChanged.connect(self.compute_and_draw_phase)

        layout = vStack(
            d_sl_elem,
            eps_sl_elem,
            iter_sl_elem,
            self.canvas
        )
        self.setLayout(layout)

        self.compute_and_draw_phase()

    def compute_time_series(self):
        return self.sys.compute(
            self.queue,
            skip=0,
            iter=self.iter_slider.value(),
            eps=self.eps_slider.value(),
            D=self.d_slider.value(),
            drive={
                "x": 0.2,
                "y": 0.1,
                "a": 1.4,
                "b": 0.3
            },
            response={
                "x": 0.02,
                "y": 0.01,
                "a": 1.4,
                "b": 0.3
            }
        )

    def compute_and_draw_phase(self, *_):
        d, r = self.compute_time_series()
        # print(d)
        self.figure.clear()
        self.ax = self.figure.subplots(2, 1)
        self.figure.tight_layout()
        self.ax[0].clear()
        self.ax[0].scatter(d.T[0], d.T[1], s=1)
        self.ax[1].clear()
        self.ax[1].scatter(r.T[0], r.T[1], s=1)
        self.canvas.draw()


if __name__ == '__main__':
    App().run()