from functools import reduce

from PyQt5.QtWidgets import QLabel, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import linregress

from common2 import *
from dynsys import SimpleApp, vStack, createSlider, ParameterizedImageWidget
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


def pipeline_each(data, fns):
    return reduce(lambda a, x: list(map(x, a)), fns, data)


def hausdorf_D(x_array, ax):
    xmin = x_array.min()
    xmax = x_array.max()

    def do_delta_and_N(box_number):
        xdelta = (xmax - xmin) / box_number
        box_indexes_of_x_array = ((x_array - xmin) / xdelta).astype(numpy.int32)
        return xdelta, len(Counter(box_indexes_of_x_array))

    box_number = numpy.array([2**i for i in range(1, 10)])  # numpy.arange(10, 1000, 10)

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

    return d, d1, abs(d2)


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("Correlation & Hausdorff")
        self.lm = LogMap(self.ctx, (900, 256))
        self.cr = Dim1D(self.ctx)

        self.figure = Figure((18, 12))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(1, 1)
        self.figure.tight_layout(pad=5)

        self.bif_tree = ParameterizedImageWidget((1, 2, 0, 1), ("lambda", None), (True, False))
        self.bif_tree.setValue((L_CRITICAL, 0))
        self.bif_tree.valueChanged.connect(self.compute)

        self.info_label = QLabel()

        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["corr", "haus"])
        self.mode_cb.currentIndexChanged.connect(self.compute)

        layout = vStack(
            self.bif_tree, self.info_label, self.canvas, self.mode_cb
        )

        self.setLayout(layout)

        self.draw_bif_tree()
        self.compute(mode="corr")

    def draw_bif_tree(self):
        self.bif_tree.setImage(self.lm.compute(
            self.queue, skip=1024, iter=512, l_min=1, l_max=2
        ))

    def compute_corr_dim(self, *_):
        l = self.get_l_value()

        r_values = numpy.linspace(0, 1.5, 100)
        res = self.lm.sample(self.queue, skip=1024, iter=2048, x=0, l=l)

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

    def compute(self, *_, mode=None):
        mode = mode if mode is not None else self.mode_cb.currentText()

        if mode == "corr":
            self.compute_corr_dim()
        elif mode == "haus":
            self.compute_hausdorff_dim()
        else:
            raise RuntimeError("no such mode: '{}'".format(mode))

    def get_l_value(self):
        return self.bif_tree.value()[0]


if __name__ == '__main__':
    App().run()
