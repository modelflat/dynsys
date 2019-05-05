from common import *
from dynsys import allocateImage, ParameterizedImageWidget, SimpleApp, createSlider, hStack, vStack
from dynsys.ui.ImageWidgets import *


SOURCE = r"""

// Henon map -- hard-coded
inline void do_step(system_t* d, system_t* r, system_t* a, param_t* p) {
    val_t d_, r_, a_;
    
    #define _F(v, a) (1 - a*v.x*v.x + v.y)
    
    d_.x = _F(d->v, d->a);
    d_.y = d->b * d->v.x;
    
    r_.x = _F(r->v, r->a) + p->eps * (_F(d->v, r->a) - _F(r->v, r->a));
    r_.y = r->b * r->v.x;
    
    a_.x = _F(a->v, a->a) + p->eps * (_F(d->v, a->a) - _F(a->v, a->a));
    a_.y = a->b * a->v.x;
    
    d->v = d_; r->v = r_; a->v = a_;
}

kernel void assist_system(
    const int skip,
    const int iter,
    
    const global system_t* d_,
    const global system_t* r_,
    const global system_t* a_,
    const global param_t* p_,
    
    global val_t* out_d,
    global val_t* out_r,
    global val_t* out_a
) {
    const int id = get_global_id(0);
    
    system_t d = d_[id], r = r_[id], a = a_[id];
    param_t p = p_[id];
    
    for (int i = 0; i < skip; ++i) {
        do_step(&d, &r, &a, &p);
    }
    
    out_d += id;
    out_r += id;
    out_a += id;
    for (int i = 0; i < iter; ++i) {
        { // output
            out_d[i] = d.v;
            out_r[i] = r.v;
            out_a[i] = a.v;
        }
        do_step(&d, &r, &a, &p);
    }
}

#define real double

inline void write_to_img(real x, real y, const float4 bounds, write_only image2d_t image) {
    int2 img = get_image_dim(image);
    img -= 1;
    real _x = (x - bounds.s0) / (bounds.s1 - bounds.s0); 
    real _y = (y - bounds.s2) / (bounds.s3 - bounds.s2);
    int2 coord = convert_int2_rtz((float2)(
        _x * img.x,
        (1.0 - _y) * img.y
    ));
    if (coord.x >= 0 && coord.y < img.x && coord.y >= 0 && coord.y < img.y) {
        write_imagef(image, coord, (float4)(0.0, 0.0, 0.0, 1.0));
    }
}

kernel void render(
    const float4 bounds_XX,
    const float4 bounds_YY,
    const global val_t* d,
    const global val_t* r,
    const global val_t* a,
    
    write_only image2d_t synchr_XX,
    write_only image2d_t synchr_YY,
    write_only image2d_t synchr_ZZ
) {
    const int id = get_global_id(0);
    // d += id;
    // r += id;
    // a += id;
    
    //printf("%f %f %f %f\n", x_r, y_r, x_a, y_a);
    
    write_to_img(d[id].x, d[id].y, bounds_XX, synchr_XX);
    write_to_img(r[id].x, r[id].y, bounds_XX, synchr_YY);
    write_to_img(a[id].x, a[id].y, bounds_XX, synchr_ZZ);
}
"""


space_shape = (-10, -10, 10, 10)


class AssistantSystem:

    def __init__(self, ctx, image_shape):
        self.image_shape = image_shape
        self.image_host_XX, self.image_dev_XX = allocateImage(ctx, image_shape)
        self.image_host_YY, self.image_dev_YY = allocateImage(ctx, image_shape)
        self.image_host_ZZ, self.image_dev_ZZ = allocateImage(ctx, image_shape)

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

    def clear(self, queue, color=(1.0, 1.0, 1.0, 1.0)):
        for img in [self.image_dev_XX, self.image_dev_YY, self.image_dev_ZZ]:
            cl.enqueue_fill_image(
                queue, img,
                color=numpy.array(color, dtype=numpy.float32),
                origin=(0,)*len(self.image_shape), region=self.image_shape
            )

    def read_from_device(self, queue):
        for img_host, img_dev in zip(
                [self.image_host_XX, self.image_host_YY, self.image_host_ZZ],
                [self.image_dev_XX, self.image_dev_YY, self.image_dev_ZZ]
        ):
            cl.enqueue_copy(
                queue, img_host, img_dev,
                origin=(0,)*len(self.image_shape), region=self.image_shape
            )
        return self.image_host_XX, self.image_host_YY, self.image_host_ZZ

    def _call_compute_time_series(self, queue, skip, iter, drives, responses, assistants, params):
        """
        kernel void assist_system(
            const int skip,
            const int iter,

            const global system_t* d_,
            const global system_t* r_,
            const global system_t* a_,
            const global param_t* p_,

            global val_t* out_d,
            global val_t* out_r,
            global val_t* out_a,
        )
        """
        assert len(drives) == len(responses) == len(assistants) == len(params)
        n = len(drives)

        drives_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=drives)
        responses_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=responses)
        assistants_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                   hostbuf=assistants)
        params_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=params)

        out = numpy.empty((3, n, iter), dtype=self.val_t)
        out_d_dev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=out[0].nbytes)
        out_r_dev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=out[1].nbytes)
        out_a_dev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=out[2].nbytes)

        self.prg.assist_system(
            queue, (n,), None,
            numpy.int32(skip),
            numpy.int32(iter),
            drives_dev, responses_dev, assistants_dev, params_dev,
            out_d_dev, out_r_dev, out_a_dev
        )

        cl.enqueue_copy(queue, out[0], out_d_dev)
        cl.enqueue_copy(queue, out[1], out_r_dev)
        cl.enqueue_copy(queue, out[2], out_a_dev)

        return out, (out_d_dev, out_r_dev, out_a_dev)

    def _call_render(self, queue, n, iter, bounds_XX, bounds_YY, d, r, a):
        """
        kernel void render(
            const float4 bounds_XX,
            const float4 bounds_YY,
            const global val_t* d,
            const global val_t* r,
            const global val_t* a,

            write_only image2d_t synchr_XX,
            write_only image2d_t synchr_YY
        );
        """
        self.clear(queue)

        self.prg.render(
            queue, (iter,), None,
            bounds_XX,
            bounds_YY,
            d, r, a,
            self.image_dev_XX,
            self.image_dev_YY,
            self.image_dev_ZZ
        )

        self.read_from_device(queue)

    def compute_and_render(self, queue, skip, iter, drives, responses, assistants, params):
        host, dev = self._call_compute_time_series(
            queue, skip, iter, drives, responses, assistants, params
        )

        min_x = numpy.amin(host["x"])
        max_x = numpy.amax(host["x"])
        min_y = numpy.amin(host["y"])
        max_y = numpy.amax(host["y"])

        bounds_XX = numpy.array((min_x, max_x, min_y, max_y), dtype=numpy.float32)
        bounds_YY = numpy.array((min_x, max_x, min_y, max_y), dtype=numpy.float32)

        self._call_render(queue, host.shape[1], iter, bounds_XX, bounds_YY, *dev)

        return self.image_host_XX, self.image_host_YY, self.image_host_ZZ


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("2.1")
        self.helper = AssistantSystem(self.ctx, image_shape=(256, 256))
        self.figure = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(self.figure)

        self.eps_slider, eps_slider_ui = createSlider(
            "r", (0, 1), withLabel="eps = {:.3f}", labelPosition="top",
            withValue=0.5
        )
        self.b_slider, b_slider_ui = createSlider(
            "r", (0.25, 0.35), withLabel="b = {:.3f}", labelPosition="top",
            withValue=0.3
        )

        self.image_XX_wgt = Image2D()
        self.image_YY_wgt = Image2D()
        self.image_ZZ_wgt = Image2D()

        self.setLayout(
            vStack(
                eps_slider_ui, b_slider_ui,
                self.canvas,
                hStack(
                    self.image_XX_wgt, self.image_YY_wgt, self.image_ZZ_wgt
                )
            )
        )

        self.connect_everything()
        self.compute()

    def connect_everything(self):
        self.eps_slider.valueChanged.connect(self.compute)
        self.b_slider.valueChanged.connect(self.compute)

    def compute(self, *_):
        t = time.perf_counter()

        ab_drv = (1.4, 0.3)
        ab_rsp = (1.4, self.b_slider.value())

        init = (0.1, 0.1)

        img_XX, img_YY, img_ZZ = self.helper.compute_and_render(
            self.queue,
            skip=0,
            iter=1 << 14,
            drives=numpy.array(
                [
                    (init, *ab_drv)
                ], dtype=self.helper.system_t
            ),
            responses=numpy.array(
                [
                    (init, *ab_rsp)
                ], dtype=self.helper.system_t
            ),
            assistants=numpy.array(
                [
                    (init, *ab_rsp)
                ], dtype=self.helper.system_t
            ),
            params=numpy.array(
                [
                    (self.eps_slider.value(),)
                ], dtype=self.helper.param_t
            )
        )

        self.image_XX_wgt.setTexture(img_XX)
        self.image_YY_wgt.setTexture(img_YY)
        self.image_ZZ_wgt.setTexture(img_ZZ)
        t = time.perf_counter() - t

        #print("{:.3f} s compute".format(t))


if __name__ == '__main__':
    App().run()