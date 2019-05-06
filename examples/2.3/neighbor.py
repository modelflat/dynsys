import numpy as np
from math import *
import random
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCharts import *
import sys
from utils import qt_layout, qt_splitter

point_size = 3.0


def henon(state, lam):
    a, b_drive, b_response = 1.4, 0.3, 0.31

    def f(x_1, x_2):
        return 1 - a * (x_1**2) + x_2

    def drive(drive_p, response_p):
        return (
            f(drive_p[0], drive_p[1]),
            b_drive * drive_p[0]
        )

    def response(drive_p, response_p):
        return (
            f(response_p[0], response_p[1]) + lam * (f(drive_p[0], drive_p[1]) - f(response_p[0], response_p[1])),
            b_response * response_p[0]
        )

    return (
        drive(*state),
        response(*state)
    )


def generate(
        skip: int,
        count: int,
        lam: float
) -> (np.ndarray, np.ndarray):

    drive_arr, response_arr = np.empty(shape=(count, 2)), np.empty(shape=(count, 2))

    drive_p, response_p = (0.1, 0.1), (0.1, 0.2)

    # skip
    for i in range(skip):
        drive_p, response_p = henon((drive_p, response_p), lam)

    # generate
    for idx in range(count):
        drive_p, response_p = henon((drive_p, response_p), lam)

        drive_arr[idx] = drive_p
        response_arr[idx] = response_p

    return (
        drive_arr,
        response_arr
    )


def show_preview(
        drive_arr,
        response_arr,
        title: str = None,
        neighbor_idxs: list = None
):
    window = QWidget()
    #
    title_lbl: QLabel = None
    if title is not None:
        title_lbl = QLabel("<center><h2>{}</h2></center>".format(title))
        title_lbl.setFixedHeight(title_lbl.sizeHint().height() + 5)
    drive_plot = QtCharts.QChart()
    drive_plot_view = QtCharts.QChartView(drive_plot)
    drive_plot_view.setRenderHint(QPainter.Antialiasing)
    drive_plot.setTitle("<h2>Drive system</h2>")
    drive_plot.createDefaultAxes()
    drive_plot.legend().setVisible(False)
    #
    response_plot = QtCharts.QChart()
    response_plot_view = QtCharts.QChartView(response_plot)
    response_plot_view.setRenderHint(QPainter.Antialiasing)
    response_plot.setTitle("<h2>Response system</h2>")
    response_plot.createDefaultAxes()
    response_plot.legend().setVisible(False)
    #
    window.setLayout(
        qt_layout(
            title_lbl,
            qt_splitter(
                drive_plot_view,
                response_plot_view,
                orientation=Qt.Horizontal
            )
        )
    )

    # draw
    drive_series = QtCharts.QScatterSeries()
    drive_series.setUseOpenGL(True)
    drive_series.setMarkerSize(point_size)
    for p in drive_arr:
        drive_series.append(*p)
    drive_plot.addSeries(drive_series)
    #
    if neighbor_idxs is not None:
        drive_neighbor_series = QtCharts.QScatterSeries()
        drive_neighbor_series.setUseOpenGL(True)
        drive_neighbor_series.setMarkerSize(point_size*2)
        for idx in neighbor_idxs:
            drive_neighbor_series.append(*drive_arr[idx])
        drive_plot.addSeries(drive_neighbor_series)
        drive_neighbor_series.setColor(Qt.red)
    #
    drive_plot.createDefaultAxes()

    response_series = QtCharts.QScatterSeries()
    response_series.setUseOpenGL(True)
    response_series.setMarkerSize(point_size)
    for p in response_arr:
        response_series.append(*p)
    response_plot.addSeries(response_series)
    #
    if neighbor_idxs is not None:
        response_neighbor_series = QtCharts.QScatterSeries()
        response_neighbor_series.setUseOpenGL(True)
        response_neighbor_series.setMarkerSize(point_size*2)
        for idx in neighbor_idxs:
            response_neighbor_series.append(*response_arr[idx])
        response_plot.addSeries(response_neighbor_series)
        response_neighbor_series.setColor(Qt.red)

    #
    response_plot.createDefaultAxes()

    return window


def D(
        drive_arr: np.ndarray,
        response_arr: np.ndarray,
        eps: float,
        support_idx: int
):
    def delta(data: np.ndarray):
        res = 0.0
        for i in range(len(data)):
            res += sqrt((data[i][0] - data[support_idx][0]) ** 2 + (data[i][1] - data[support_idx][1]) ** 2)
        return 1 / len(data) * res

    drive_neighbor_idxs = list()
    for idx in range(len(drive_arr)):
        if sqrt(
            (drive_arr[support_idx][0] - drive_arr[idx][0])**2 +
            (drive_arr[support_idx][1] - drive_arr[idx][1])**2
        ) <= eps:
            drive_neighbor_idxs.append(idx)

    d = 0
    for idx in drive_neighbor_idxs:
        d += sqrt(
            (response_arr[support_idx][0] - response_arr[idx][0])**2 +
            (response_arr[support_idx][1] - response_arr[idx][1])**2
        )
    return (
        drive_neighbor_idxs,
        1 / (len(drive_neighbor_idxs) * delta(response_arr)) * d
    )


if __name__ == '__main__':
    app = QApplication(sys.argv)

    eps = 0.025
    skip = 0
    count = 8192*4
    lam_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]

    d_arr = list()

    window = QTabWidget()
    support_idx = random.randrange(count)
    for lam in lam_list:
        drive_arr, response_arr = generate(skip, count, lam)

        neighbor_idxs, d = D(drive_arr, response_arr, eps, support_idx)

        d_arr.append((lam, d))

        tab = show_preview(drive_arr, response_arr, "\u03bb = {} | Henon".format(lam), neighbor_idxs)

        window.addTab(tab, "\u03bb = {}".format(lam))
        print("\u03bb = {} was computed".format(lam))

    d_plot = QtCharts.QChart()
    d_plot_view = QtCharts.QChartView(d_plot)
    d_plot_view.setRenderHint(QPainter.Antialiasing)
    d_plot.setTitle("<h2>Мера</h2>")
    d_plot.createDefaultAxes()
    d_plot.legend().setVisible(False)
    #
    d_series = QtCharts.QLineSeries()
    d_series.setUseOpenGL(True)
    for d_p in d_arr:
        d_series.append(*d_p)
        print("{0:0<5}\t{1:0<20}".format(*d_p))
    d_plot.addSeries(d_series)
    d_plot.createDefaultAxes()
    d_plot.axisX().setTitleText("\u03bb")
    d_plot.axisY().setTitleText("d")
    #
    window.addTab(d_plot_view, "Мера")

    window.showMaximized()
    app.exec_()




