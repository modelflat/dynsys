import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox, QFrame
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from common2 import *
from dynsys import SimpleApp, allocateImage, vStack, hStack, ParameterizedImageWidget, createSlider

PARAM_MAP_SOURCE = r"""
#define real double
#define TWOPI (2 * M_PI)

inline real map(real x, real om, real k) { 
    return fmod(x + om + k * sin(TWOPI * x) / TWOPI, 1.0);
}

kernel void compute_parameter_map(
    const real om_min, const real om_max,
    const real k_min, const real k_max,
    const int skip, const int iter,
    real x,
    global real* res
) {
    const int2 id = (int2)(get_global_id(0), get_global_id(1));
    
    const real om = om_min + id.x * (om_max - om_min) / get_global_size(0);
    const real k = k_min + id.y * (k_max - k_min) / get_global_size(1);
    
    for (int i = 0; i < skip; ++i) {
        x = map(x, om, k);
    }
    
    res += (id.x + id.y * get_global_size(0)) * iter;
    
    for (int i = 0; i < iter; ++i) {
        res[i] = x;
        x = map(x, om, k);
    }
}

void make_heap(global real*, int, int);
void make_heap(global real* data, int n, int i) {
    while (true) {
        int smallest = i;
        int l = (i << 1) + 1;
        int r = (i << 1) + 2;

        if (l < n && data[l] > data[smallest]) {
            smallest = l;
        }
        if (r < n && data[r] > data[smallest]) {
            smallest = r;
        }
        if (smallest == i) {
            return; // already smallest
        }

        real t = *(data + i); *(data + i) = *(data + smallest); *(data + smallest) = t;

        i = smallest;
    }
}

void heap_sort(global real*, int);
void heap_sort(global real* data, int n) {
    for (int i = n / 2 - 1; i >= 0; --i) {
        make_heap(data, n, i);
    }

    for (int i = n - 1; i >= 0; --i) {
        real t = *(data); *(data) = *(data + i); *(data + i) = t;
        make_heap(data, i, 0);
    }
}

inline int count_unique(global real* data, real precision, int n) {
    heap_sort(data, n);
    real prev = data[0];
    int uniques = 1;
    
    for (int i = 1; i < n; ++i) {
        real next = data[i];
        if (fabs(prev - next) > precision) {
            prev = next;
            ++uniques;
        }
    }
    
    return uniques;
}

inline float3 color_for_count(int count, int total) {
    if (count == total) {
        return 0.25;
    }
    const float d = 1.0 / count * 8;
    switch(count % 8) {
        case 1:
            return (float3)(1.0, 0.0, 0.0)*d;
        case 2:
            return (float3)(0.0, 1.0, 0.0)*d;
        case 3:
            return (float3)(0.0, 0.0, 1.0)*d;
        case 4:
            return (float3)(1.0, 0.0, 1.0)*d;
        case 5:
            return (float3)(1.0, 1.0, 0.0)*d;
        case 6:
            return (float3)(0.0, 1.0, 1.0)*d;
        case 7:
            return (float3)(0.5, 0.0, 0.0)*d;
        default:
            return count == 8 ? .5 : d;
    }
}

kernel void draw_parameter_map(
    const int iter, const real precision,
    global real* res,
    global int* period,
    write_only image2d_t out
) {
    const int2 id = (int2)(get_global_id(0), get_global_id(1));
    res += (id.x + id.y * get_global_size(0)) * iter;
    
    int uniques = count_unique(res, precision, iter);
    
    period[id.x + (get_global_size(1) - id.y - 1) * get_global_size(0)] = uniques;
    
    float3 color = color_for_count(uniques, iter);
    
    write_imagef(out, (int2)(id.x, get_image_height(out) - 1 - id.y), (float4)(color, 1.0));
}

"""


LYAP_MAP = r"""

inline real d_map(real x, real om, real k) {
    return 1 + k * cos(TWOPI * x);
}

real lyap(const int iter, real x, real om, real k) {
    real L = 0;
    for (int i = 0; i < iter; ++i) {
        L += log(fabs(d_map(x, om, k)));
        x = map(x, om, k);
    }
    return L / iter;
}

float3 color_for_lyap(const real l) {
    if      (l > 0) {
        const real k = sqrt(l);
        return (float3)(k, 0.5*k, 0.0);
    }
    else if (l < 0) {
        const real k = sqrt(fabs(l));
        return (float3)(0.0, 0.5*k, k);
    }
    else { // == 0
        return 1.0;
    } 
}

kernel void compute_lyap(
    const int skip, const int iter,
    real x,
    const global real* om_,
    const global real* k_,
    global real* l
) {
    const int id = get_global_id(0);
    
    const real om = om_[id];
    const real k = k_[id];

    for (int i = 0; i < skip; ++i) {
        x = map(x, om, k);
    }
    
    l[id] = lyap(iter, x, om, k);
}


kernel void compute_lyap_map(
    const real om_min, const real om_max,
    const real k_min, const real k_max,
    const int skip, const int iter,
    real x,
    global real* res,
    write_only image2d_t img
) {
    const int2 id = (int2)(get_global_id(0), get_global_id(1));
    
    const real om = om_min + id.x * (om_max - om_min) / get_global_size(0);
    const real k = k_min + id.y * (k_max - k_min) / get_global_size(1);
    
    for (int i = 0; i < skip; ++i) {
        x = map(x, om, k);
    }
    
    const real v = res[id.x + id.y * get_global_size(0)] = lyap(iter, x, om, k);
    
    float3 color = color_for_lyap(v);
    
    write_imagef(img, (int2)(id.x, get_image_height(img) - 1 - id.y), (float4)(color, 1.0));
}

"""


ITER_DIAG_AND_UTILS = r"""

#define real3 double3
#define real4 double4

kernel void sample(
    const int skip, const int iter,
    real x, 
    const global real* om_,
    const global real* k_,
    global real* res
) {
    const int id = get_global_id(0);
    const real om = om_[id];
    const real k = k_[id];
    
    res += id * iter;
    
    for (int i = 0; i < skip; ++i) {
        x = map(x, om, k);
    }
    
    for (int i = 0; i < iter; ++i) {
        res[i] = x;
        x = map(x, om, k);
    }
}

kernel void draw_lines(
    const real om, const real k, 
    const real4 bounds,
    write_only image2d_t result
) {
    const int2 id = (int2)(get_global_id(0), get_global_id(1));
    
    const real x = bounds.s0 + id.x * (bounds.s1 - bounds.s0) / get_global_size(0);
    const real y = bounds.s2 + (get_global_size(1) - id.y - 1) * (bounds.s3 - bounds.s2) / get_global_size(1);
    
    if (fabs(y - x) < 1e-3) {
        write_imagef(result, id, (float4)(1.0, 0.0, 0.0, 1.0));
    } else if (fabs(y - map(x, om, k)) < 2e-3) {
        write_imagef(result, id, (float4)(0.0, 0.5, 0.0, 1.0));
    }
}

kernel void draw_cobweb(
    const global real* samples,
    const real4 bounds,
    write_only image2d_t result
) {
    const int id = get_global_id(0);
    
    if (id + 2u >= get_global_size(0)) {
        return;
    }
    
    const real3 x = (real3)(samples[id], samples[id+1], samples[id+2]);

    if (isnan(x.s0) || isnan(x.s1) || isnan(x.s2)) {
        return;
    }
    
    const int2 im = get_image_dim(result);
    
    int2 p1 = {
        convert_int_rtz((x.s0 - bounds.s0) / (bounds.s1 - bounds.s0) * im.x),
        im.y - convert_int_rtz((x.s1 - bounds.s2) / (bounds.s3 - bounds.s2) * im.y) - 1
    };
    p1 = clamp(p1, (int2)(0, 0), (int2)(im - 1));
    
    int2 p2 = {
        convert_int_rtz((x.s1 - bounds.s0) / (bounds.s1 - bounds.s0) * im.x),
        im.y - convert_int_rtz((x.s2 - bounds.s2) / (bounds.s3 - bounds.s2) * im.y) - 1
    };
    p2 = clamp(p2, (int2)(0, 0), (int2)(im - 1));
    
    int2 line;
     
    line = (int2)(min(p1.x, p2.x), max(p1.x, p2.x));
    for (int i = line.s0; i <= line.s1; ++i) {
        write_imagef(result, (int2)(i, p1.y), (float4)(0.0, 0.0, 0.0, 1.0));
    }
    
    line = (int2)(min(p1.y, p2.y), max(p1.y, p2.y));
    for (int i = line.s0; i <= line.s1; ++i) {
        write_imagef(result, (int2)(p2.x, i), (float4)(0.0, 0.0, 0.0, 1.0));
    }
}

"""


ROTATION = r"""

inline real map_no_fmod(real x, real om, real k) {
    return x + om + k * sin(TWOPI * x) / TWOPI;
}

kernel void compute_rotation_no(
    const int iter,
    const real x0,
    const global real* om_,
    const global real* k_,
    global real* res
) {
    const int id = get_global_id(0);
    
    const real om = om_[id];
    const real k = k_[id];
    
    real x = x0;
    
    for (int i = 0; i < iter; ++i) {
        x = map_no_fmod(x, om, k);
    }
    
    res[id] = (x - x0) / iter;
}

"""


# TODO why this number?
GOLDEN = 0.6066610634702
INV_OMEGA = 1 / GOLDEN

DELTA_1 = -2.833610655891167799
DELTA_2 =  1.660424381098700680


def rescale(point, om_min, om_max, l_min, l_max):
    om_min = point - (point - om_min) / DELTA_1
    om_max = point + (om_max - point) / DELTA_1
    l_min, l_max= l_min / INV_OMEGA, l_max / INV_OMEGA
    return om_min, om_max, l_min, l_max


def rescale_n_times(n, point, om_min, om_max, l_min, l_max):
    for _ in range(n):
        om_min, om_max, l_min, l_max = rescale(point, om_min, om_max, l_min, l_max)
    return om_min, om_max, l_min, l_max


def fib(n):
    assert n >= 2
    a = numpy.empty((n,), dtype=numpy.int64)
    a[0] = 0
    a[1] = 1
    for i in range(2, n):
        a[i] = a[i - 1] + a[i - 2]
    return a


class CircleMap:

    def __init__(self, ctx, shape, iter_diag_shape):
        self.ctx = ctx
        self.prg = cl.Program(ctx, "\n".join([
            PARAM_MAP_SOURCE, LYAP_MAP, ITER_DIAG_AND_UTILS, ROTATION
        ])).build()
        self.shape = shape
        self.iter_diag_shape = iter_diag_shape
        self.img = allocateImage(ctx, shape)
        self.iter_diag_img = allocateImage(ctx, iter_diag_shape)

    def draw_param_map(self, queue, skip, iter, om_min, om_max, k_min, k_max, x0=0, precision=1e-2):
        """
        kernel void compute_parameter_map(
            const real om_min, const real om_max,
            const real k_min, const real k_max,
            const int skip, const int iter,
            const real x, const int decimals,
            global int* res
        )
        """
        clear_image(queue, self.img[1], self.shape)

        res_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=8 * iter * numpy.prod(self.shape))

        self.prg.compute_parameter_map(
            queue, self.shape, None,
            numpy.float64(om_min),
            numpy.float64(om_max),
            numpy.float64(k_min),
            numpy.float64(k_max),
            numpy.int32(skip), numpy.int32(iter),
            numpy.float64(x0),
            res_dev
        )

        periods = numpy.empty(self.shape, numpy.int32)
        periods_dev = alloc_like(self.ctx, periods)

        self.prg.draw_parameter_map(
            queue, self.shape, None,
            numpy.int32(iter), numpy.float64(precision),
            res_dev,
            periods_dev,
            self.img[1]
        )

        cl.enqueue_copy(queue, periods, periods_dev)

        return read(queue, *self.img, self.shape), periods

    def draw_lyap_map(self, queue, skip, iter, om_min, om_max, k_min, k_max, x0=0):
        """
        kernel void compute_lyap_map(
            const real om_min, const real om_max,
            const real k_min, const real k_max,
            const int skip, const int iter,
            real x,
            global real* res
        )
        """
        clear_image(queue, self.img[1], self.shape)

        lyap = numpy.empty(self.shape, dtype=numpy.float64)
        lyap_dev = alloc_like(self.ctx, lyap)

        self.prg.compute_lyap_map(
            queue, self.shape, None,
            numpy.float64(om_min),
            numpy.float64(om_max),
            numpy.float64(k_min),
            numpy.float64(k_max),
            numpy.int32(skip), numpy.int32(iter),
            numpy.float64(x0),
            lyap_dev,
            self.img[1]
        )

        cl.enqueue_copy(queue, lyap, lyap_dev)

        return read(queue, *self.img, self.shape), lyap

    def draw_iter_diag(self, queue, skip, iter, bounds, om, k, x0=0):
        assert len(bounds) == 4

        om_buf = numpy.array([om], numpy.float64)
        om_buf_dev = copy_dev(self.ctx, om_buf)
        k_buf = numpy.array([k], numpy.float64)
        k_buf_dev = copy_dev(self.ctx, k_buf)

        res_dev = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=8 * iter)

        clear_image(queue, self.iter_diag_img[1], self.iter_diag_shape)

        queue.finish()

        self.prg.sample(
            queue, (1,), None,
            numpy.int32(skip), numpy.int32(iter),
            numpy.float64(x0),
            om_buf_dev, k_buf_dev,
            res_dev
        )

        self.prg.draw_lines(
            queue, self.iter_diag_shape, None,
            numpy.float64(om), numpy.float64(k),
            numpy.array(bounds, dtype=numpy.float64),
            self.iter_diag_img[1]
        )

        self.prg.draw_cobweb(
            queue, (iter,), None,
            res_dev,
            numpy.array(bounds, dtype=numpy.float64),
            self.iter_diag_img[1]
        )

        return read(queue, *self.iter_diag_img, self.iter_diag_shape)

    def compute_many_lyaps(self, queue, skip, iter, om_values, k_values, x0=0):
        assert len(om_values) == len(k_values)

        om_buf_dev = copy_dev(self.ctx, numpy.array(om_values, dtype=numpy.float64))
        k_buf_dev = copy_dev(self.ctx, numpy.array(k_values, dtype=numpy.float64))

        l_buf = numpy.empty((len(om_values), ), dtype=numpy.float64)
        l_buf_dev = alloc_like(self.ctx, l_buf)

        self.prg.compute_lyap(
            queue, (len(om_values),), None,
            numpy.int32(skip), numpy.int32(iter),
            numpy.float64(x0),
            om_buf_dev, k_buf_dev,
            l_buf_dev
        )

        cl.enqueue_copy(queue, l_buf, l_buf_dev)

        return l_buf

    def compute_many_rotation_nums(self, queue, iter, om_values, k_values, x0=0):
        assert len(om_values) == len(k_values)

        om_dev = copy_dev(self.ctx, om_values)
        k_dev = copy_dev(self.ctx, k_values)
        res = numpy.empty(len(om_values), dtype=numpy.float64)
        res_dev = alloc_like(self.ctx, res)

        self.prg.compute_rotation_no(
            queue, (len(om_values), ), None,
            numpy.int32(iter),
            numpy.float64(x0),
            om_dev, k_dev, res_dev
        )

        cl.enqueue_copy(queue, res, res_dev)

        return res


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("3")

        self.cm = CircleMap(self.ctx, (600, 480), (512, 512))

        self.left_wgt = ParameterizedImageWidget((0, 1, 0, 4), ("om", "k"),
                                                 (True, True), targetColor=Qt.white)
        self.right_wgt = ParameterizedImageWidget((0, 0, 0, 0), (None, None), shape=(False, False))

        self.period_map = None
        self.lyap_map = None

        self.period_label = QLabel()
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["param", "lyap", "show both", "3.b", "staircase"])
        self.mode_cb.currentTextChanged.connect(self.switch_mode)

        self.left_wgt.valueChanged.connect(self.draw_iter_diag)

        self.figure = Figure((18, 12))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(1, 1)
        self.figure.tight_layout(pad=6)
        self.canvas_frame = QFrame()

        self.staircase_k_touched = False

        self.scale_slider, scale_slider_wgt = createSlider(
            "i", (0, 8), withValue=0, withLabel="scale = {}", labelPosition="top"
        )

        self.canvas_frame.setLayout(vStack(
            scale_slider_wgt,
            self.canvas
        ))

        self.canvas_frame.hide()

        layout = vStack(
            self.mode_cb,
            vStack(
                self.period_label,
                hStack(
                    self.left_wgt,
                    vStack(self.canvas_frame, self.right_wgt)
                )
            )
        )
        self.setLayout(layout)

        self.draw_param_map()
        self.draw_iter_diag((0.35, 0.9))

    def plot_staircase(self, *_, k=1):
        n = 1 << 10

        if self.staircase_k_touched:
            k = self.left_wgt.value()[1]

        iter = 1 << 14

        om_min, om_max, w_min, w_max = 0, 1, 0, 1

        scale_times = self.scale_slider.value()

        om_min, om_max, w_min, w_max = rescale_n_times(scale_times, GOLDEN, om_min, om_max, w_min, w_max)

        om_values = numpy.linspace(om_min, om_max, n)
        k_values = numpy.full(n, k, numpy.float64)

        res = self.cm.compute_many_rotation_nums(self.queue, iter, om_values, k_values)

        fib_ns = fib(16)

        closest_points = []
        labels = []
        for n1, n2 in zip(fib_ns[::], fib_ns[1::]):
            golden_approx = n1 / n2
            # print("{} / {} ~=".format(n1, n2), golden_approx)
            # closest = numpy.argmin(numpy.abs(res - golden_approx))
            closest = numpy.argmin(numpy.abs(res - golden_approx))
            closest_points.append(closest)
            # labels.extend(["{}/{}".format(n1, n2)]*len(closest))
            labels.append("{}/{}".format(n1, n2))

        closest_points = numpy.array(closest_points)

        self.ax.clear()
        # self.ax.set_xlim(om_min, om_max)
        # self.ax.set_ylim(w_min, w_max)
        self.ax.plot(om_values, res)

        i = 0
        for label, x, y in zip(labels, om_values[closest_points], res[closest_points]):
            self.ax.annotate(
                label,
                xy=(x, y), xytext=(-10 * i if i % 2 == 0 else 10 * i, 10 * i if i % 2 == 0 else -10 * i),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
            i += 1

        self.canvas.draw()

    def plot_lyap_along_k_1(self, *_):
        n = 1 << 10
        k = 1
        om_min, om_max, l_min, l_max = 0, 1, -2, 0

        scale_times = self.scale_slider.value()

        om_min, om_max, l_min, l_max = rescale_n_times(
            scale_times, GOLDEN, om_min, om_max, l_min, l_max
        )

        om_values = numpy.linspace(om_min, om_max, n)

        lyaps = self.cm.compute_many_lyaps(
            self.queue, 1 << 10, 1 << 12, om_values, numpy.full(n, k, dtype=numpy.float64)
        )

        self.ax.clear()
        self.ax.set_xlim(om_min, om_max)
        self.ax.set_ylim(l_min, l_max)
        self.ax.plot(om_values, lyaps)

        self.canvas.draw()

    def draw_iter_diag(self, param):
        om, k = param
        t = time.perf_counter()
        texture = self.cm.draw_iter_diag(
            self.queue,
            skip=1 << 9, iter=1 << 7,
            bounds=(0, 1, 0, 1),
            om=om, k=k
        )
        self.queue.finish()
        t = time.perf_counter() - t

        # print("Iter diag computed in {:.3f} s".format(t))
        self.right_wgt.setImage(texture)

    def draw_lyap_map(self, *_, img=None):
        import time

        t = time.perf_counter()
        texture, lyap_values = self.cm.draw_lyap_map(
            self.queue,
            skip=1 << 9, iter=1 << 7,
            om_min=0,
            om_max=1,
            k_min=0,
            k_max=4,
        )
        t = time.perf_counter() - t

        print("Lyapunov map computed in {:.3f} s ".format(t))
        if img is None:
            self.left_wgt.setImage(texture.copy())
        else:
            img.setImage(texture.copy())

    def draw_param_map(self, *_):
        import time

        t = time.perf_counter()
        texture, period_map = self.cm.draw_param_map(
            self.queue,
            skip=1 << 9, iter=1 << 7,
            om_min=0,
            om_max=1,
            k_min=0,
            k_max=4,
            precision=1e-2
        )
        t = time.perf_counter() - t
        print("Parameter map computed in {:.3f} s ".format(t))
        self.period_map = period_map
        self.left_wgt.setImage(texture.copy())

    def switch_mode(self, mode=None):
        mode = self.mode_cb.currentText() if mode is None else mode

        print("switching to '{}'".format(mode))

        self.left_wgt.setShape((True, True))
        try:
            self.left_wgt.valueChanged.disconnect()
            self.scale_slider.valueChanged.disconnect()
        except:
            print("[error] disconnect failed")

        if mode == "param":
            self.canvas_frame.hide()
            self.right_wgt.setVisible(True)
            self.left_wgt.valueChanged.connect(self.draw_iter_diag)

            self.draw_param_map()
        elif mode == "lyap":
            self.canvas_frame.hide()
            self.right_wgt.setVisible(True)
            self.left_wgt.valueChanged.connect(self.draw_iter_diag)

            self.draw_lyap_map()
        elif mode == "show both":
            self.canvas_frame.hide()
            self.right_wgt.setVisible(True)

            self.draw_param_map()
            self.draw_lyap_map(img=self.right_wgt)
        elif mode == "3.b":
            self.canvas_frame.show()
            self.left_wgt.setShape((False, True))
            self.right_wgt.setVisible(False)
            self.scale_slider.valueChanged.connect(self.plot_lyap_along_k_1)

            self.plot_lyap_along_k_1()
        elif mode == "staircase":
            self.canvas_frame.show()
            self.left_wgt.setShape((False, True))
            self.right_wgt.setVisible(False)
            self.scale_slider.valueChanged.connect(self.plot_staircase)

            def _k_preserving(value):
                self.staircase_k_touched = True
                self.plot_staircase(value)

            self.left_wgt.valueChanged.connect(_k_preserving)

            self.draw_param_map()
            self.staircase_k_touched = False
            self.plot_staircase(k=1)
        else:
            raise RuntimeError("No such mode: '{}'".format(mode))


if __name__ == '__main__':
    App().run()
