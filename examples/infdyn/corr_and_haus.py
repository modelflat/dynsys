from functools import reduce

from PyQt5.QtWidgets import QLabel, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import linregress

from common2 import *
from dynsys import SimpleApp, vStack, createSlider
from t1 import LogMap, L_CRITICAL
from collections import Counter


CORR_KERNEL = r"""
// simple correlation kernel, blows up your cl device with global mem and atomic accesses 
kernel void corr(
    const int n, 
    const global double* r_, 
    const global double* arr,
    global int* counts
) {
    const int id = get_global_id(0);
    const int r_id = get_global_id(1);
    
    const double r = r_[r_id];
    const double p = arr[id];
    
    int count = 0;
    for (int i = 0; i < n; ++i) {
        count += (fabs(arr[i] - p) <= r);
    }
    
    atomic_add(counts + r_id, count);
}
"""


class Dim1D:

    def __init__(self, ctx):
        self.ctx = ctx
        src = [
            CORR_KERNEL
        ]
        self.prg = cl.Program(ctx, "\n".join(src)).build()

    def corr(self, queue, arr, r_values):
        n = len(arr)
        n_r = len(r_values)

        arr_dev = copy_dev(self.ctx, arr)
        r_values_dev = copy_dev(self.ctx, r_values)

        res = numpy.zeros(n_r, numpy.int32)
        res_dev = copy_dev(self.ctx, res)
        self.prg.corr(
            queue, (n, n_r), None,
            numpy.int32(n),
            r_values_dev, arr_dev, res_dev
        )
        cl.enqueue_copy(queue, res, res_dev)

        return res / n ** 2

    def hausdorf(self, queue, arr):
        arr_min, arr_max = arr.min(), arr.max()


def pipeline_each(data, fns):
    return reduce(lambda a, x: list(map(x, a)), fns, data)


def hausdorf_D(x_array, ax):
    xmin = x_array.min()
    xmax = x_array.max()

    def do_delta_and_N(box_number):
        xdelta = (xmax - xmin) / box_number
        box_indexes_of_x_array = ((x_array - xmin) / xdelta).astype(numpy.int32)
        return xdelta, len(Counter(box_indexes_of_x_array))

    box_number = numpy.arange(10, 1000, 10)

    delta, n_non_empty = numpy.array(list(map(do_delta_and_N, box_number))).T

    ax.clear()
    ax.plot(delta, n_non_empty, color="b", linestyle="-", marker=".")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()

    delta = numpy.log(delta)
    n_non_empty = numpy.log(n_non_empty)

    d = -n_non_empty.sum() / delta.sum()
    d1 = (n_non_empty.max() - n_non_empty.min()) / (delta.max() - delta.min())
    d2 = linregress(delta, n_non_empty).slope

    return d, d1, d2


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("Correlation & Hausdorff")
        self.lm = LogMap(self.ctx, (1, 1))
        self.cr = Dim1D(self.ctx)

        self.figure = Figure((18, 12))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(1, 1)
        self.figure.tight_layout(pad=4)

        self.l_slider, l_slider_el = createSlider("r", (1, 2), withValue=L_CRITICAL)
        self.l_slider.valueChanged.connect(self.compute_corr_dim)

        self.info_label = QLabel()
        self.l_label = QLabel()
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["corr", "haus"])

        self.mode_cb.currentIndexChanged.connect(self.switch_mode)

        layout = vStack(
            self.mode_cb,
            self.l_label,
            l_slider_el,
            self.info_label,
            self.canvas
        )

        self.setLayout(layout)

        self.switch_mode("corr")

    def switch_mode(self, *_, mode=None):
        mode = mode if mode is not None else self.mode_cb.currentText()

        self.l_slider.valueChanged.disconnect()

        print("switching to {}".format(mode))

        if mode == "corr":
            self.l_slider.valueChanged.connect(self.compute_corr_dim)
            self.compute_corr_dim()
        elif mode == "haus":
            self.l_slider.valueChanged.connect(self.compute_hausdorff_dim)
            self.compute_hausdorff_dim()
        else:
            raise RuntimeError("no such mode: '{}'".format(mode))

    def compute_corr_dim(self, *_):
        l = self.get_l_value()

        r_values = numpy.linspace(0, 1.5, 100)
        res = self.lm.sample(self.queue, skip=200, iter=2000, x=0, l=l)

        corrs = self.cr.corr(self.queue, res, r_values)

        D = corrs.sum() / r_values.sum()
        D1 = (corrs.max() - corrs.min()) / (r_values.max() - r_values.min())
        D2 = linregress(r_values, corrs).slope

        self.info_label.setText(
            "D corr average: {:.5f}\nD corr max average: {:.5f}\nD corr regress: {:.5f}".format(D, D1, D2)
        )

        self.ax.clear()
        self.ax.plot(r_values, corrs, color="b", linestyle="-", marker=".")
        self.ax.set_yscale("log")
        self.ax.set_xscale("log")
        self.ax.grid()

        self.canvas.draw()

    def compute_hausdorff_dim(self, *_):
        l = self.get_l_value()

        res = self.lm.sample(self.queue, skip=200, iter=2000, x=0, l=l)

        D, D1, D2 = hausdorf_D(res, self.ax)

        self.info_label.setText(
            "D haus average: {:.5f}\nD haus max average: {:.5f}\nD haus regress: {:.5f}".format(D, D1, D2)
        )

        self.canvas.draw()

    def get_l_value(self):
        l = self.l_slider.value()
        self.l_label.setText("lambda = {}".format(l))
        return l


if __name__ == '__main__':
    App().run()
