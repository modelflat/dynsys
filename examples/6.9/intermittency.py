import numpy as np
from math import *
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCharts import *
import sys
from utils import qt_layout, qt_splitter

point_size = 3.0


def henon(state, lam):
    a, b_drive, b_response = 1.4, 0.3, 0.3

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

    drive_p, response_p = (0.1, 0.1), (0.5, 0.5)

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


def draw(
        intermittency_arr: np.ndarray,
        x_distribution: np.ndarray = None,
        y_distribution: np.ndarray = None,
        title: str = None
):
    title_lbl = None
    if title is not None:
        title_lbl = QLabel("<center><h2>{}</h2></center>".format(title))
        title_lbl.setFixedHeight(title_lbl.sizeHint().height() + 5)

    x_plot = QtCharts.QChart()
    x_plot_view = QtCharts.QChartView(x_plot)
    x_plot_view.setRenderHint(QPainter.Antialiasing)
    x_plot.createDefaultAxes()
    x_plot.legend().setVisible(False)
    x_plot.setTitle("<h1>X</h1>")
    #
    y_plot = QtCharts.QChart()
    y_plot_view = QtCharts.QChartView(y_plot)
    y_plot_view.setRenderHint(QPainter.Antialiasing)
    y_plot.createDefaultAxes()
    y_plot.legend().setVisible(False)
    y_plot.setTitle("<h1>Y</h1>")
    #
    x_series = QtCharts.QLineSeries()
    # x_series = QtCharts.QScatterSeries()
    # x_series.setMarkerSize(point_size)
    x_series.setUseOpenGL(True)
    for idx, p in enumerate(intermittency_arr):
        x_series.append(idx, p[0])
    x_plot.addSeries(x_series)
    x_plot.createDefaultAxes()
    x_plot.axisX().setTitleText("Time (points)")
    x_plot.axisY().setTitleText("|X<sup>d</sup> - X<sup>r</sup>|")
    #
    y_series = QtCharts.QLineSeries()
    # y_series = QtCharts.QScatterSeries()
    # y_series.setMarkerSize(point_size)
    y_series.setUseOpenGL(True)
    for idx, p in enumerate(intermittency_arr):
        y_series.append(idx, p[1])
    y_plot.addSeries(y_series)
    y_plot.createDefaultAxes()
    y_plot.axisX().setTitleText("Time (points)")
    y_plot.axisY().setTitleText("|Y<sup>d</sup> - Y<sup>r</sup>|")

    xy_plot = QtCharts.QChart()
    xy_plot_view = QtCharts.QChartView(xy_plot)
    xy_plot_view.setRenderHint(QPainter.Antialiasing)
    xy_plot.createDefaultAxes()
    xy_plot.legend().setVisible(False)
    #
    xy_series = QtCharts.QScatterSeries()
    xy_series.setMarkerSize(point_size)
    xy_series.setUseOpenGL(True)
    for p in intermittency_arr:
        xy_series.append(*p)
    xy_plot.addSeries(xy_series)
    xy_plot.createDefaultAxes()
    xy_plot.axisX().setTitleText("|X<sup>d</sup> - X<sup>r</sup>|")
    xy_plot.axisY().setTitleText("|Y<sup>d</sup> - Y<sup>r</sup>|")

    x_distr_plot: QtCharts.QChart = None
    x_distr_plot_view: QtCharts.QChartView = None
    if x_distribution is not None:
        x_distr_plot = QtCharts.QChart()
        x_distr_plot_view = QtCharts.QChartView(x_distr_plot)
        x_distr_plot_view.setRenderHint(QPainter.Antialiasing)
        x_distr_plot.createDefaultAxes()
        x_distr_plot.legend().setVisible(False)
        x_distr_plot.setTitle("<h1>X distribution</h1>")
        #
        x_distr_series = QtCharts.QBarSeries()
        x_distr_series.setUseOpenGL(True)
        labels = list()
        x_distr_set = QtCharts.QBarSet("")
        for p in x_distribution:
            x_distr_set.append(p[1])
            labels.append(str(p[0]))
        x_distr_series.append(x_distr_set)

        x_distr_plot.addSeries(x_distr_series)
        x_distr_plot.createDefaultAxes()
        x_axis = QtCharts.QBarCategoryAxis()
        x_axis.setMinorGridLineVisible(False)
        x_axis.append(labels)
        x_distr_plot.setAxisX(x_axis, x_distr_series)
        x_distr_plot.axisX().setTitleText("Duration")
        x_distr_plot.axisY().setTitleText("Entries")

    y_distr_plot: QtCharts.QChart = None
    y_distr_plot_view: QtCharts.QChartView = None
    if y_distribution is not None:
        y_distr_plot = QtCharts.QChart()
        y_distr_plot_view = QtCharts.QChartView(y_distr_plot)
        y_distr_plot_view.setRenderHint(QPainter.Antialiasing)
        y_distr_plot.createDefaultAxes()
        y_distr_plot.legend().setVisible(False)
        y_distr_plot.setTitle("<h1>Y distribution</h1>")
        #
        y_distr_series = QtCharts.QBarSeries()
        y_distr_series.setUseOpenGL(True)
        labels = list()
        y_distr_set = QtCharts.QBarSet("")
        for p in y_distribution:
            y_distr_set.append(p[1])
            labels.append(str(p[0]))
        y_distr_series.append(y_distr_set)

        y_distr_plot.addSeries(y_distr_series)
        y_distr_plot.createDefaultAxes()
        y_axis = QtCharts.QBarCategoryAxis()
        y_axis.setMinorGridLineVisible(False)
        y_axis.append(labels)
        y_distr_plot.setAxisX(y_axis, y_distr_series)
        y_distr_plot.axisX().setTitleText("Duration")
        y_distr_plot.axisY().setTitleText("Entries")

    widget = QWidget()
    widget.setLayout(
        qt_layout(
            title_lbl,
            qt_splitter(
                qt_splitter(
                    qt_splitter(x_plot_view, x_distr_plot_view, orientation=Qt.Horizontal),
                    qt_splitter(y_plot_view, y_distr_plot_view, orientation=Qt.Horizontal)
                ),
                xy_plot_view
            )
        )
    )

    return widget


def detect_phases(
        arr: np.ndarray,
        deviation_percent: float = 0.1
):
    # Work only with intermittency_arr with absolute difference
    # 0 - ламинарная
    # 1 - турбулентная
    eps = arr.max() * deviation_percent

    dur = np.zeros_like(arr, dtype=np.int8)
    for idx, p in enumerate(arr):
        if p > eps:        # p <= 0.0 + eps
            dur[idx] = 1

    return dur
    # return arr > eps


def duration_distribution(
        phases: np.ndarray
):
    distr = dict()

    prev = phases[0]
    counter = 0
    idx = 1
    while idx < len(phases):
        if phases[idx] == 0:
            if prev == phases[idx]:
                counter += 1
            else:
                counter = 1
        elif prev != phases[idx]:
            if counter in distr:
                distr[counter] += 1
            else:
                distr[counter] = 1
        prev = phases[idx]
        idx += 1
    else:
        if counter in distr:
            distr[counter] += 1
        else:
            distr[counter] = 1

    distr_arr = np.empty(shape=(len(distr), 2))
    for idx, i in enumerate(distr.items()):
        distr_arr[idx] = i

    return distr_arr


if __name__ == '__main__':
    app = QApplication(sys.argv)

    lam_critical = 0.365

    eps = 0.025
    skip = 256
    count = 1024
    # count = 8192
    lam_list = [0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.285, 0.29, 0.295,
                0.30, 0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345,
                0.35, 0.355, 0.36, lam_critical, 0.37, 0.375, 0.38, 0.385, 0.39, 0.395, 0.40]

    window = QTabWidget()

    x_mean_duration = list()
    y_mean_duration = list()

    for lam in lam_list:
        drive_arr, response_arr = generate(skip, count, lam)

        intermittency_arr = np.empty(shape=(len(drive_arr), 2))
        for idx in range(len(intermittency_arr)):
            intermittency_arr[idx] = (
                drive_arr[idx][0] - response_arr[idx][0],
                drive_arr[idx][1] - response_arr[idx][1]
            )

        x_phases = detect_phases(abs(intermittency_arr[:, 0]))
        y_phases = detect_phases(abs(intermittency_arr[:, 1]))

        x_distr = duration_distribution(x_phases)
        y_distr = duration_distribution(y_phases)

        x_mean_duration.append((
            lam_critical - lam,
            x_distr[:, 0].sum() / x_distr[:, 1].sum()
        ))

        y_mean_duration.append((
            lam_critical - lam,
            y_distr[:, 0].sum() / y_distr[:, 1].sum()
        ))

        tab = draw(
            # abs(intermittency_arr),
            intermittency_arr,
            x_distr,
            y_distr,
            title="\u03bb = {} | \u03bb<sub>cr</sub> = {} | \u0394 \u03bb = {}".format(lam, lam_critical, lam_critical - lam)
        )
        window.addTab(tab, "\u03bb = {}".format(lam))

        print("\u03bb = {} was computed".format(lam))

    plot = QtCharts.QChart()
    plot_view = QtCharts.QChartView(plot)
    plot_view.setRenderHint(QPainter.Antialiasing)
    plot.createDefaultAxes()
    plot.legend().setVisible(True)
    #
    x_series = QtCharts.QLineSeries()
    x_series.setUseOpenGL(True)
    x_series.setName("X")
    for p in x_mean_duration:
        x_series.append(*p)
    plot.addSeries(x_series)
    #
    y_series = QtCharts.QLineSeries()
    y_series.setUseOpenGL(True)
    y_series.setName("Y")
    for p in y_mean_duration:
        y_series.append(*p)
    plot.addSeries(y_series)
    #
    plot.createDefaultAxes()
    plot.axisX().setTitleText("\u0394 \u03bb (\u03bb<sub>cr</sub> - \u03bb)")
    plot.axisY().setTitleText("Средняя длительность ламинарных фаз (в точках)")
    #
    window.addTab(plot_view, "Sum")

    print("x_mean_duration", x_mean_duration)
    print("y_mean_duration", y_mean_duration)

    window.showMaximized()
    app.exec_()
