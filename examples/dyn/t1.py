from common2 import *
from dynsys import SimpleApp, allocateImage, Image2D, vStack, hStack, createSlider
from PyQt5.QtWidgets import QLabel, QComboBox, QFrame

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


L_CRITICAL = 1.40115_51890_92050_6
DELTA = 4.66920_16091_02990_7
SCALE_ALPHA = -2.50290_78750_95892_8


def rescale_log_map(l_min, l_max, x_min, x_max):
    l_min = L_CRITICAL - (L_CRITICAL - l_min) / DELTA
    l_max = L_CRITICAL + (l_max - L_CRITICAL) / DELTA
    x_min, x_max = x_min / SCALE_ALPHA, x_max / SCALE_ALPHA
    return l_min, l_max, x_min, x_max


def rescale_n_times(n, l_min, l_max, x_min, x_max):
    for _ in range(n):
        l_min, l_max, x_min, x_max = rescale_log_map(l_min, l_max, x_min, x_max)
    return l_min, l_max, x_min, x_max


BIFTREE_SOURCE = r"""
#define real double

inline real map(real x, real l) {
    return 1 - l * x * x;
}

kernel void bif_tree(
    const int skip, const int iter,
    real x, const real l_min, const real l_max,
    global real* out
) {
    const int id = get_global_id(0);
    const real l = l_min + id * (l_max - l_min) / get_global_size(0);
    
    for (int i = 0; i < skip; ++i) {
        x = map(x, l);
    }
    
    out += iter * id;
    
    for (int i = 0; i < iter; ++i) {
        out[i] = x; 
        x = map(x, l);
    }
}

inline void write_point(
    const real x, const real x_min, const real x_max, const int l_id,
    const int flip_y,
    write_only image2d_t out
) {
    if (x > x_max || x < x_min || isnan(x)) return;
    const int h = get_image_height(out) - 1;
    
    int x_coord = convert_int_rtz((x - x_min) / (x_max - x_min) * h);
    
    write_imagef(out, (int2)(l_id, (flip_y) ? h - x_coord : x_coord), (float4)(0.0, 0.0, 0.0, 1.0));
}

kernel void draw_bif_tree(
    const int iter,
    const real x_min, const real x_max,
    const int flip_y,
    const global real* data,
    write_only image2d_t img
) {
    const int id = get_global_id(0);
    data += id * iter;
    for (int i = 0; i < iter; ++i) {
        write_point(data[i], x_min, x_max, id, flip_y, img);
    }
}

// --- 

kernel void sample(
    const int skip, const int iter, real x, real l,
    global real* res
) {
    for (int i = 0; i < skip; ++i) {
        x = map(x, l);
    }
    
    for (int i = 0; i < iter; ++i) {
        res[i] = x;
        x = map(x, l);
    }
}

kernel void sample_n(
    const int skip, const int iter, real x,
    const global real* l,
    global real* res
) {
    const int id = get_global_id(0);
    sample(skip, iter, x, l[id], res + id * iter);
}

/*
kernel void detect_phase(
    const int iter, const int round_dec,
    const int win_size, const int turbulence_threshold,
    const global real* d,
    global int* p 
) {
    const int id = get_global_id(0);
    d += id * iter;
    p += id * iter;
    
    const int i = get_global_id(1);
    const int l = max(0, i - win_size / 2);
    const int r = min(iter, i + win_size / 2);
    for (int j = l; j < r; ++j) {
        const int encoded = convert_int_rtz(d[j] * pow(10, round_dec));
    }
}
*/

"""


LYAP_SOURCE = r"""

inline real d_map(real x, real l) {
    return - 2*l*x;
}

real log_map_lyap(const int iter, real x, real l) {
    real L = 0;
    for (int i = 0; i < iter; ++i) {
        L += log(fabs(d_map(x, l)));
        x = map(x, l);
    }
    return L / iter;
}

kernel void lyap(
    const int skip, const int iter,
    real x, const real l_min, const real l_max,
    global real* L
) {
    const int id = get_global_id(0);
    const real l = l_min + id * (l_max - l_min) / get_global_size(0);
    
    for (int i = 0; i < skip; ++i) {
        x = map(x, l);
    }
    
    L[id] = log_map_lyap(iter, x, l);
}

"""


class LogMap:

    def __init__(self, ctx, shape):
        self.ctx = ctx
        self.shape = shape
        self.img = allocateImage(ctx, shape)
        sources = [
            BIFTREE_SOURCE, LYAP_SOURCE
        ]
        self.prg = cl.Program(ctx, "\n".join(sources)).build()

    def compute_multiple_lambdas(self, queue, skip, iter, l_min, l_max, x0=0):
        buf = numpy.empty((self.shape[0], iter), dtype=numpy.float64)
        buf_dev = alloc_like(self.ctx, buf)
        if l_min == l_max:
            return buf, buf_dev
        self.prg.bif_tree(
            queue, (self.shape[0],), None,
            numpy.int32(skip), numpy.int32(iter),
            numpy.float64(x0), numpy.float64(l_min), numpy.float64(l_max),
            buf_dev
        )
        cl.enqueue_copy(queue, buf, buf_dev)
        return buf, buf_dev

    def compute_lyap(self, queue, skip, iter, l_min, l_max, n, x0=0):
        buf = numpy.empty((n,), dtype=numpy.float64)
        buf_dev = alloc_like(self.ctx, buf)
        self.prg.lyap(
            queue, (n,), None,
            numpy.int32(skip), numpy.int32(iter),
            numpy.float64(x0),
            numpy.float64(l_min), numpy.float64(l_max),
            buf_dev
        )
        cl.enqueue_copy(queue, buf, buf_dev)
        return buf

    def sample(self, queue, skip, iter, x, l):
        buf = numpy.empty((iter,), dtype=numpy.float64)
        buf_dev = alloc_like(self.ctx, buf)
        self.prg.sample(
            queue, (1,), None,
            numpy.int32(skip), numpy.int32(iter),
            numpy.float64(x), numpy.float64(l),
            buf_dev
        )
        cl.enqueue_copy(queue, buf, buf_dev)
        return buf

    def _call_draw(self, queue, iter, buf_dev, x_min, x_max):
        clear_image(queue, self.img[1], self.shape)
        if x_min == x_max:
            return self.img[0]

        flip_y = 1 if x_min < x_max else 0
        if x_min > x_max:
            x_min, x_max = x_max, x_min

        self.prg.draw_bif_tree(
            queue, (self.shape[0],), None,
            numpy.int32(iter),
            numpy.float64(x_min), numpy.float64(x_max),
            numpy.int32(flip_y),
            buf_dev,
            self.img[1]
        )
        read(queue, *self.img, self.shape)
        return self.img[0]

    def compute(self, queue, skip, iter, l_min, l_max, x0=0, x_min=-8, x_max=8):
        buf, buf_dev = self.compute_multiple_lambdas(queue, skip, iter, l_min, l_max, x0)
        x_min = max(x_min, numpy.amin(buf))
        x_max = min(x_max, numpy.amax(buf))
        return self._call_draw(queue, iter, buf_dev, x_min, x_max)


class Scaling(SimpleApp):

    def __init__(self):
        super(Scaling, self).__init__("1.a / 1.c")
        self.lm = LogMap(self.ctx, (1200, 600))
        self.im = Image2D(targetShape=(True, False))

        self.l_min_slider, l_min_slider_el = createSlider("r", bounds=(0.75, 2), withLabel="l_min = {}",
                                                    labelPosition="top", withValue=1)

        self.l_max_slider, l_max_slider_el = createSlider("r", bounds=(0.75, 2), withLabel="l_max = {}",
                                                    labelPosition="top", withValue=2)

        self.x_min_slider, x_min_slider_el = createSlider("r", bounds=(-1, 1), withLabel="x_min = {}",
                                                    labelPosition="top", withValue=-1)

        self.x_max_slider, x_max_slider_el = createSlider("r", bounds=(-1, 1), withLabel="x_max = {}",
                                                    labelPosition="top", withValue=1)

        self.iter_slider, iter_slider_el = createSlider("i", bounds=(4, 1024), withLabel="iter = {}",
                                                        labelPosition="top")

        self.scale_slider, scale_slider_el = createSlider("i", bounds=(0, 8), withLabel="scale = {}",
                                                          withValue=0,
                                                          labelPosition="top")

        self.mode_cb = QComboBox()
        self.mode_cb.addItem("bif")
        self.mode_cb.addItem("lyap")

        self.info_label = QLabel()

        self.figure = Figure((18, 12))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(1, 1)
        self.figure.tight_layout(pad=2)

        self.canvas_frame = QFrame()
        self.canvas_frame.setLayout(hStack(self.canvas))

        layout = vStack(
            hStack(self.info_label, self.mode_cb),
            hStack(iter_slider_el, scale_slider_el),
            hStack(l_min_slider_el, l_max_slider_el),
            hStack(x_min_slider_el, x_max_slider_el),
            hStack(self.im, self.canvas_frame)
        )

        self.l_min_slider.valueChanged.connect(self.scale)
        self.l_max_slider.valueChanged.connect(self.scale)
        self.x_min_slider.valueChanged.connect(self.scale)
        self.x_max_slider.valueChanged.connect(self.scale)
        self.mode_cb.currentIndexChanged.connect(self.scale)
        # self.iter_slider.valueChanged.connect(self.compute_bif_tree)

        self.scale_slider.valueChanged.connect(self.scale)

        self.setLayout(layout)
        self.scale()

    def compute_bt(self, l_min, l_max, x_min, x_max):
        self.canvas_frame.hide()
        self.im.setVisible(True)
        img = self.lm.compute(self.queue, 1 << 10, 1 << 10,
                              l_min=l_min,
                              l_max=l_max,
                              x_min=x_min,
                              x_max=x_max)
        self.im.setTexture(img)

    def compute_lyap(self, l_min, l_max):
        self.canvas_frame.show()
        self.im.setVisible(False)

        n = 1 << 10

        lyap = self.lm.compute_lyap(self.queue, 1 << 10, 1 << 14,
                                    l_min=l_min, l_max=l_max, n=n)

        self.ax.clear()

        self.ax.plot(numpy.linspace(l_min, l_max, n), lyap)
        self.ax.axhline(0, color="black", linestyle="--")

        self.canvas.draw()

    def scale(self, *_):
        l_min = self.l_min_slider.value()
        l_max = self.l_max_slider.value()
        x_min = self.x_min_slider.value()
        x_max = self.x_max_slider.value()
        scale_times = self.scale_slider.value()

        if (l_min > L_CRITICAL or l_max < L_CRITICAL) and scale_times != 0:
            print("bad lambda bounds")
            return

        l_min, l_max, x_min, x_max = rescale_n_times(scale_times, l_min, l_max, x_min, x_max)

        self.info_label.setText("l_min = {}\t\tl_max = {}\t\tx_min = {}\t\tx_max = {}".format(
            l_min, l_max, x_min, x_max
        ))

        mode = self.mode_cb.currentText()

        if mode == "bif":
            self.compute_bt(l_min, l_max, x_min, x_max)
        elif mode == "lyap":
            self.compute_lyap(l_min, l_max)
        else:
            raise RuntimeError("unsupported mode '{}'".format(mode))


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("1.b / 1.d")


if __name__ == '__main__':
    # Scaling().run()
    App().run()
