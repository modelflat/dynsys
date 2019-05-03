from dynsys import *
from dynsys.ui.SliderWidgets import createSlider
from dynsys.LCE import dummyOption
import pyopencl as cl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import time

SOURCE = r"""
#define real double

#define N 2
#define _F(x1, x2, a) (1 - a*x1*x1 + x2)

inline void map(real d[N], real r[N], real a[N], const real params_d[N], const real params_r[N], const real eps) {
    const real x_d = d[0];
    const real y_d = d[1];
    const real x_r = r[0];
    const real y_r = r[1];
    const real x_a = a[0];
    const real y_a = a[1];
    const real a_d = params_d[0];
    const real b_d = params_d[1];
    const real a_r = params_r[0];
    const real b_r = params_r[1];
    
    d[0] = _F(x_d, y_d, a_d);
    d[1] = b_d * x_d;
    
    r[0] = _F(x_r, y_r, a_r) + eps * (_F(x_d, y_d, a_r) - _F(x_r, y_r, a_r));
    r[1] = b_r * x_r;
    
    a[0] = _F(x_a, y_a, a_r) + eps * (_F(x_d, y_d, a_r) - _F(x_a, y_a, a_r));
    a[1] = b_r * x_a;    
}

kernel void helper(
    const int skip, 
    const int iter,
    const real eps,
    const global real* params,
    const global real* initial,
    global real* out
) {
    const int id = get_global_id(0);
    initial += 4 * id;
    
    real x_d[N];
    real x_r[N];
    real x_a[N];
    real p_d[N];
    real p_r[N];
    {
        x_d[0] = initial[0]; 
        x_d[1] = initial[1];
        x_r[0] = initial[0];
        x_r[1] = initial[1];
        x_a[0] = initial[2];
        x_a[1] = initial[3];
        p_d[0] = params[0]; 
        p_d[1] = params[1];
        p_r[0] = params[2];
        p_r[1] = params[3];
    }
    
    for (int i = 0; i < skip; ++i) {
        map(x_d, x_r, x_a, p_d, p_r, eps);
    }
    
    out += 6 * iter * id;
    for (int i = 0; i < iter; ++i) {
        out[6*i + 0] = x_d[0];
        out[6*i + 1] = x_d[1];
        out[6*i + 2] = x_r[0];
        out[6*i + 3] = x_r[1];
        out[6*i + 4] = x_a[0];
        out[6*i + 5] = x_a[1];
        map(x_d, x_r, x_a, p_d, p_r, eps);
    }
}

inline void write_to_img(real x, real y, const float4 bounds, write_only image2d_t image) {
    int2 img = get_image_dim(image);
    img -= 1;
    real _x = (x - bounds.s0) / (bounds.s1 - bounds.s0); 
    real _y = (y - bounds.s2) / (bounds.s3 - bounds.s2);
    int2 coord = convert_int2_rtz((float2)(
        _x * img.x,
        (1.0 - _y) * img.y
    ));
    write_imagef(image, coord, (float4)(0.0, 0.0, 0.0, 1.0));
}

kernel void render(
    const float4 bounds_XX,
    const float4 bounds_YY,
    const global real* result,
    write_only image2d_t synchr_XX,
    write_only image2d_t synchr_YY
) {
    const int id = get_global_id(0);
    result += 6 * id;
    
    real x_r = result[2];
    real y_r = result[3];
    real x_a = result[4];
    real y_a = result[5];
    
    //printf("%f %f %f %f\n", x_r, y_r, x_a, y_a);
    
    write_to_img(x_r, y_r, bounds_XX, synchr_XX);
    write_to_img(y_r, y_a, bounds_YY, synchr_YY);
}


"""


space_shape = (-10, -10, 10, 10)


class HelperSystem:

    def __init__(self, ctx, image_shape):
        self.image_shape = image_shape
        self.image_host_XX, self.image_dev_XX = allocateImage(ctx, image_shape)
        self.image_host_YY, self.image_dev_YY = allocateImage(ctx, image_shape)
        # self.image_dev_ZZ = allocateImage(ctx, image_shape)
        self.prg = cl.Program(ctx, SOURCE).build()
        self.ctx = ctx

    def clear(self, queue, color=(1.0, 1.0, 1.0, 1.0)):
        cl.enqueue_fill_image(
            queue, self.image_dev_XX,
            color=numpy.array(color, dtype=numpy.float32),
            origin=(0,)*len(self.image_shape), region=self.image_shape
        )
        cl.enqueue_fill_image(
            queue, self.image_dev_YY,
            color=numpy.array(color, dtype=numpy.float32),
            origin=(0,)*len(self.image_shape), region=self.image_shape
        )

    def read_from_device(self, queue):
        cl.enqueue_copy(
            queue, self.image_host_XX, self.image_dev_XX,
            origin=(0,)*len(self.image_shape), region=self.image_shape
        )
        cl.enqueue_copy(
            queue, self.image_host_YY, self.image_dev_YY,
            origin=(0,)*len(self.image_shape), region=self.image_shape
        )
        return self.image_host_XX, self.image_host_YY

    def __call__(self, queue, initial, initial_assist, params, eps, skip, iter):
        n = min(len(initial), len(initial_assist))
        init = numpy.empty((n, 4), dtype=numpy.float64)
        for i, vals in enumerate(zip(initial, initial_assist)):
            init[i, :2], init[i, 2:4] = vals

        init_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=init)

        n_params = min(len(p) for p in params)
        params_host = numpy.empty((len(params), n_params), dtype=numpy.float64)

        for i, param_set in enumerate(params):
            params_host[i, :] = [p for _, p in sorted(param_set.items(), key=lambda x: x[0])]

        params_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=params_host)

        out = numpy.empty((n, iter, 6), dtype=numpy.float64)
        out_dev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=out.nbytes)

        self.prg.helper(
            queue, (n,), None,
            numpy.int32(skip),
            numpy.int32(iter),
            numpy.float64(eps),
            params_dev,
            init_dev,
            out_dev
        )

        cl.enqueue_copy(queue, out, out_dev)

        self.clear(queue)

        _mins = numpy.amin(out, axis=1)[0][2:]
        _maxs = numpy.amax(out, axis=1)[0][2:]

        bounds_XX = numpy.array((_mins[0], _maxs[0], _mins[2], _maxs[2]), dtype=numpy.float32)
        bounds_YY = numpy.array((_mins[1], _maxs[1], _mins[3], _maxs[3]), dtype=numpy.float32)

        print(bounds_XX)
        print(bounds_YY)

        self.prg.render(
            queue, (iter,), None,
            bounds_XX,
            bounds_YY,
            out_dev,
            self.image_dev_XX,
            self.image_dev_YY
        )

        self.read_from_device(queue)

        return out, self.image_host_XX, self.image_host_YY


class Task2_1(SimpleApp):

    def __init__(self):
        super(Task2_1, self).__init__("2.1")
        self.helper = HelperSystem(self.ctx, image_shape=(480, 320))
        self.figure = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(self.figure)

        special = numpy.ones((256, 256, 4), dtype=numpy.uint8)
        special[:, :, :] = 255
        special.T[2] = 0

        self.init_sel = ParameterizedImageWidget(
            bounds=(-1, 1, -1, 1),
            names=("x", "y"),
            shape=(256, 256), textureShape=(256, 256)
        )
        self.init_sel.setImage(special)
        self.init_assist_sel = ParameterizedImageWidget(
            bounds=(-1, 1, -1, 1),
            names=("x", "y"),
            shape=(256, 256), textureShape=(256, 256)
        )
        self.init_assist_sel.setImage(special)
        self.param_sel = ParameterizedImageWidget(
            bounds=(0, 2, 0.25, 0.35),
            names=("a", "b"),
            shape=(256, 256), textureShape=(256, 256)
        )
        self.param_sel.setImage(special)
        self.param_assist_sel = ParameterizedImageWidget(
            bounds=(0, 2, 0.25, 0.35),
            names=("a", "b"),
            shape=(256, 256), textureShape=(256, 256)
        )
        self.param_assist_sel.setImage(special)

        self.eps_slider, self.eps_slider_ui = createSlider(
            "r", (0, 1), withLabel="eps = {:.3f}", labelPosition="top",
            withValue=0.5
        )

        self.image_XX_wgt = Image2D()
        self.image_YY_wgt = Image2D()

        self.setLayout(
            hStack(
                vStack(
                    self.eps_slider_ui,
                    hStack(self.init_sel, self.init_assist_sel),
                    hStack(self.param_sel, self.param_assist_sel),
                    cm = (2, 2, 2, 2)
                ),
                vStack(self.canvas),
                vStack(
                    self.image_XX_wgt,
                    self.image_YY_wgt
                )
            )
        )

        self.connect_everything()
        self.compute()

    def connect_everything(self):
        self.param_sel.valueChanged.connect(self.compute)
        self.param_assist_sel.valueChanged.connect(self.compute)
        self.init_sel.valueChanged.connect(self.compute)
        self.init_assist_sel.valueChanged.connect(self.compute)
        self.eps_slider.valueChanged.connect(self.compute)

    def compute(self, *_):
        t = time.perf_counter()
        ab = (1.4, 0.3) #self.param_sel.value()
        ab_ass = self.param_assist_sel.value()
        init = self.init_sel.value()
        init_ass = (0.1, 0.2) #self.init_assist_sel.value()
        result, img_XX, img_YY = self.helper(
            self.queue,
            initial=(
                init,
            ),
            initial_assist=(
                init_ass,
            ),
            params=[
                {"a": ab[0], "b": ab[1]},
                {"a": ab_ass[0], "b": ab_ass[1]},
            ],
            eps=self.eps_slider.value(),
            skip=0,
            iter=1 << 14
        )
        self.image_XX_wgt.setTexture(img_XX)
        self.image_YY_wgt.setTexture(img_YY)
        t = time.perf_counter() - t

        tt = time.perf_counter()
        # self.figure.clear()
        # self.ax = self.figure.subplots(3, 1)
        # self.figure.tight_layout()
        # self.ax[0].clear()
        # self.ax[0].scatter(result.T[0], result.T[1], s=1)
        # self.ax[1].clear()
        # self.ax[1].scatter(result.T[2], result.T[4], s=1)
        # self.ax[2].clear()
        # self.ax[2].scatter(result.T[3], result.T[5], s=1)
        # self.canvas.draw()
        tt = time.perf_counter() - tt


        print("{:.3f} s compute, {:.3f} s draw".format(t, tt))




if __name__ == '__main__':
    Task2_1().run()