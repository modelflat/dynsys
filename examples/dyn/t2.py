from t1 import LogMap
from dynsys import SimpleApp, createSlider, vStack, hStack

from collections import defaultdict

import numpy

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import QLabel, QCheckBox

from tqdm import tqdm
from scipy.optimize import curve_fit


def count_phases(arr):
    lengths = defaultdict(lambda: 0)
    c_l, c_t = 0, 0
    prev = arr[0]
    boundaries = []
    c = 0
    for i in range(1, len(arr)):
        if arr[i] == prev:
            c += 1
            continue
        else:
            boundaries.append(i)
            if prev == 0:
                lengths[c] += 1
                c_l += 1
            if prev == 1:
                c_t += 1
            c = 0
        prev = arr[i]

    if prev == arr[-1]:
        if prev == 0:
            lengths[c] += 1
            c_l += 1
        if prev == 1:
            c_t += 1

    return c_l, c_t, boundaries, lengths


def detect_phases(res, round_dec=2, win_size=16, turbulence_if_gt=3):
    res = numpy.around(res, decimals=round_dec)
    n = len(res)
    phase = numpy.zeros(res.shape, dtype=numpy.int32)

    for i in range(n):
        dist = defaultdict(lambda: 0)
        for j in range(max(0, i - win_size // 2), min(n, i + win_size // 2)):
            dist[res[j]] += 1
        if len(dist) > turbulence_if_gt:
            phase[i] = 1

    phase_clean = numpy.empty_like(phase)

    # eliminate outliers
    for i in range(n):
        phase_clean[i] = 1 if numpy.average(
            phase[max(0, i - win_size // 2): min(n, i + win_size // 2)]
        ) >= 0.5 else 0

    c_l, c_t, boundaries, l_lengths = count_phases(phase_clean)

    return phase_clean, {"l": c_l, "t": c_t}, boundaries, l_lengths


class App(SimpleApp):

    def __init__(self, compute_lam_lengths):
        super(App, self).__init__("2.a")
        self.lm = LogMap(self.ctx, (1, 1))

        self.l_min, self.l_max = 1.7499, 1.75001

        self.l_slider, l_slider_el = createSlider("r", (self.l_min, self.l_max),
                                                  withLabel="l = {}",
                                                  labelPosition="top",
                                                  withValue=1.75)

        self.cached_lyap = None

        self.figure = Figure((18, 12))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(3, 1)
        self.figure.tight_layout(pad=3)

        self.lam_count = QLabel()

        layout = vStack(
            self.lam_count,
            l_slider_el,
            self.canvas
        )

        self.l_slider.valueChanged.connect(self.sample)

        self.setLayout(layout)

        self.sample()
        if compute_lam_lengths:
            self.lam_lengths()

    def call_sample(self, skip, iter, l):
        res = self.lm.sample(self.queue, skip, iter, x=0, l=l)
        phase, counts, boundaries, l_lengths = detect_phases(
            res,
            round_dec=1,
            win_size=20,
            turbulence_if_gt=3
        )
        return res, phase, counts, boundaries, l_lengths

    def sample(self, *_):
        skip = 1 << 10
        iter = 1 << 12
        n_lyap = 1 << 10
        iter_lyap = 1 << 16

        l = self.l_slider.value()
        res, phase, counts, boundaries, l_lengths = self.call_sample(skip, iter, l)

        self.lam_count.setText("Phase count: lam: {l}, tur: {t}".format(**counts))

        x = numpy.array(range(iter), dtype=numpy.int64)

        self.ax[0].clear()
        self.ax[0].scatter(x[phase == 0], res[phase == 0], color="blue", s=1)
        self.ax[0].scatter(x[phase == 1], res[phase == 1], color="red", s=1)
        for i in boundaries:
            self.ax[0].axvline(i, linestyle="--", color="black", linewidth=1)

        if self.cached_lyap is None:
            self.cached_lyap = self.lm.compute_lyap(self.queue, skip, iter_lyap, self.l_min, self.l_max, n_lyap)

        self.ax[1].clear()
        self.ax[1].plot(numpy.linspace(self.l_min, self.l_max, n_lyap), self.cached_lyap)
        self.ax[1].axvline(l, color="red", linestyle="--", linewidth=1)

        self.canvas.draw()

    def lam_lengths(self, *_):
        n = 1 << 8
        skip = 4  # 1 << 10
        iter = 1 << 13
        lambdas = numpy.linspace(self.l_min, 1.749999, n)

        av_lam = []
        for l in tqdm(lambdas):
            _, _, _, _, l_lengths = self.call_sample(skip, iter, l)
            av_lam.append(numpy.average(list(l_lengths.keys())))

        lambdas = (1.75 - lambdas)[::-1]
        av_lam = numpy.array(av_lam[::-1])

        def curve(x, a):
            return x ** -a

        p_opt, _ = curve_fit(curve, lambdas, av_lam)
        a, = -p_opt

        self.ax[2].clear()
        self.ax[2].plot(lambdas, av_lam)
        self.ax[2].plot(lambdas, lambdas ** a, label="approx. pow. value = {:.5f}".format(a))
        self.ax[2].legend()
        self.canvas.draw()


if __name__ == '__main__':
    App(False).run()
