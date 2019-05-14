from functools import reduce

from PyQt5.QtWidgets import QLabel, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import linregress

from common2 import *
from dynsys import SimpleApp, vStack, createSlider, Image2D, ParameterizedImageWidget
from t1 import LogMap, L_CRITICAL


import numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def ac_manual(x, lags):
    n_var = x.var() * len(x)
    x_center = x - x.mean()

    return numpy.array(
        [1 if lag == 0 else (x_center[lag:] * x_center[:-lag]).sum() / n_var for lag in lags],
        dtype=numpy.float64
    )


def ac_fft(x, lags):
    n_var = x.var() * len(x)
    x_center = x - x.mean()

    dft = numpy.fft.fft(x_center)
    corr = numpy.fft.ifft(dft.conjugate() * dft).real / n_var

    return corr[:len(lags)]


def ac_numpy(x, lags):
    n_var = x.var() * len(x)
    x_center = x - x.mean()

    corr = numpy.correlate(x_center, x_center, mode="full")[len(x) - 1:] / n_var

    return corr[:len(lags)]


class App(SimpleApp):

    def __init__(self):
        super(App, self).__init__("ACF & Fourier")

        self.l_min, self.l_max = 1, 2

        self.lm = LogMap(self.ctx, (900, 256))

        self.figure = Figure((18, 12))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(1, 1)

        self.l_slider, l_slider_el = createSlider("r", (self.l_min, self.l_max),
                                                  withLabel="lambda = {}",
                                                  withValue=L_CRITICAL,
                                                  labelPosition="top")

        self.l_slider.valueChanged.connect(self.compute_akf)

        self.bif_tree = ParameterizedImageWidget((self.l_min, self.l_max, 0, 1), ("lambda", None), (True, False))

        layout = vStack(
            self.bif_tree,
            l_slider_el,
            self.canvas
        )

        self.setLayout(layout)

        self.draw_bif_tree()
        self.compute_akf()

    def draw_bif_tree(self):
        self.bif_tree.setImage(self.lm.compute(
            self.queue, skip=1024, iter=512, l_min=self.l_min, l_max=self.l_max
        ))

    def compute_akf(self, *_):
        l = self.l_slider.value()

        self.bif_tree.setValue((l, 0))

        res = self.lm.sample(self.queue, skip=1024, iter=512, x=0, l=l)

        lags = range(64 + 1)

        self.ax.clear()

        autocorrs = {
            "Manual AC": (ac_manual, "-"),
            "FFT + IFFT AC": (ac_fft, "-."),
            "NumPy AC": (ac_numpy, "--"),
        }

        for lab, (f, ls) in autocorrs.items():
            a_corr = f(res, lags)
            self.ax.plot(lags, abs(a_corr), label=lab, linestyle=ls)

        self.ax.set_xlabel('lag')
        self.ax.set_ylabel('correlation coefficient')
        self.ax.legend()

        self.canvas.draw()


if __name__ == '__main__':
    App().run()

