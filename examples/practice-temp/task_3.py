from PySide2.QtCore import *
from PySide2.QtWidgets import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import sys
from random import Random

import pyopencl as cl

from dynsys.LCE import dummyOption


def vStack(*args):
    l = QVBoxLayout()
    for a in args:
        if isinstance(a, QLayout):
            l.addLayout(a)
        else:
            l.addWidget(a)
    return l


def hStack(*args):
    l = QHBoxLayout()
    for a in args:
        if isinstance(a, QLayout):
            l.addLayout(a)
        else:
            l.addWidget(a)
    return l

from dynsys.cl.Core import createContextAndQueue

blank_src = """
float2 fun(
    float2 point,
    float a, float b
) {
    //! \\todo
    return point;
}
"""

losi_src = """
float2 forward_fun(
    float2 p,
    float a, float b
) {
    float2 res;
    res.x = 1 - a*fabs(p.x) + b*p.y;
    res.y = p.x;
    return res;
}

float2 backward_fun(
    float2 p,
    float a, float b
) {
    float2 res;
    res.x = p.y;
    res.y = (p.x - 1.0 + a*(res.x))/b;
    return res;
}
"""

phase_src = """
kernel void homoclinic_plot(
    const float a,
    const float b,
    const int count,
    
    global const float2* initial_f, 
    global const float2* initial_b,
    
    global float* x_f,
    global float* y_f,
    global float* x_b,
    global float* y_b
) {
    int idx = get_global_id(0);
    
    float2 fp = initial_f[idx];
    float2 bp = initial_b[idx];
    
    for(int i = 0; idx < count; ++idx){
        x_f[idx*count+i] = fp.x;        
        y_b[idx*count+i] = fp.y;
        
        fp = forward_fun(fp, a, b);
    }
    
    for(int i = 0; idx < count; ++idx){
        x_b[idx*count+i] = bp.x;        
        y_b[idx*count+i] = bp.y;
        
        fp = backward_fun(fp, a, b);
    }
}
"""


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.figure_f = plt.figure()
        self.plot_canvas_f = FigureCanvas(self.figure_f)
        self.plot_f = self.figure_f.add_subplot(111)
        self.plot_f.set_xlabel('X')
        self.plot_f.set_ylabel('Y')
        self.plot_f.set_title("Forward time")
        self.figure_b = plt.figure()
        self.plot_canvas_b = FigureCanvas(self.figure_b)
        self.plot_b = self.figure_b.add_subplot(111)
        self.plot_b.set_xlabel('X')
        self.plot_b.set_ylabel('Y')
        self.plot_b.set_title("Backward time")

        self.a_var = QDoubleSpinBox(self)
        self.a_var.setRange(-100, 100)
        self.a_var.setDecimals(3)
        self.a_var.setSingleStep(0.1)
        self.b_var = QDoubleSpinBox(self)
        self.b_var.setRange(-100, 100)
        self.b_var.setDecimals(3)
        self.b_var.setSingleStep(0.1)

        self.stable_x = QDoubleSpinBox()
        self.stable_x.setRange(-100, 100)
        self.stable_x.setDecimals(4)
        self.stable_x.setSingleStep(0.01)
        self.stable_y = QDoubleSpinBox()
        self.stable_y.setRange(-100, 100)
        self.stable_y.setDecimals(4)
        self.stable_y.setSingleStep(0.01)

        self.forward_rad = QDoubleSpinBox(self)
        self.forward_rad.setRange(0, 100)
        self.forward_rad.setDecimals(4)
        self.forward_rad.setSingleStep(0.01)
        self.forward_rad.setValue(0.01)
        self.backward_rad = QDoubleSpinBox(self)
        self.backward_rad.setRange(0, 100)
        self.backward_rad.setDecimals(4)
        self.backward_rad.setSingleStep(0.01)
        self.backward_rad.setValue(1.0)

        self.count = QSpinBox(self)
        self.count.setRange(0, 1000)
        self.count.setSingleStep(1)
        self.count.setValue(1)
        self.marker_size = QDoubleSpinBox(self)
        self.marker_size.setRange(0, 100)
        self.marker_size.setSingleStep(0.2)
        self.marker_size.setValue(1)

        self.resolution_sb = QSpinBox(self)
        self.resolution_sb.setRange(2, 768)
        self.resolution_sb.setSingleStep(4)
        self.resolution_sb.setValue(8)

        self.connect_checker = QCheckBox("Auto-draw", self)

        self.draw_btn = QPushButton("Draw", self)

        self.equation_cb = QComboBox(self)
        self.equation_cb.addItem("", [blank_src, 0.0, 0.0])
        self.equation_cb.addItem("3.c) Losi", [losi_src, 1.4, 0.3])
        self.equation_text = QTextEdit()
        self.equation_text.setText(blank_src)

        def _connect_toggle(state):
            if state is 2:
                self.a_var.valueChanged.connect(self._draw)
                self.b_var.valueChanged.connect(self._draw)
                self.stable_x.valueChanged.connect(self._draw)
                self.stable_y.valueChanged.connect(self._draw)
            else:
                self.a_var.valueChanged.disconnect(self._draw)
                self.b_var.valueChanged.disconnect(self._draw)
                self.stable_x.valueChanged.disconnect(self._draw)
                self.stable_y.valueChanged.disconnect(self._draw)

        def _change_src():
            self.equation_text.setText(self.equation_cb.currentData()[0])
            self.a_var.setValue(self.equation_cb.currentData()[1])
            self.b_var.setValue(self.equation_cb.currentData()[2])
            # self.count.setValue(self.equation_cb.currentData()[5])
            # self.resolution_sb.setValue(self.equation_cb.currentData()[6])

        self.equation_cb.currentIndexChanged.connect(_change_src)
        self.connect_checker.stateChanged.connect(_connect_toggle)
        self.draw_btn.clicked.connect(self._draw)

        top_wdg = QWidget()
        top_wdg.setLayout(
            hStack(
                self.plot_canvas_f,
                self.plot_canvas_b
            )
        )

        bot_wgd = QWidget()
        bot_wgd.setLayout(
            hStack(
                vStack(
                    hStack(QLabel("A var:"), self.a_var),
                    hStack(QLabel("B var:"), self.b_var),
                    hStack(QLabel("Stable X/Y point:"), self.stable_x, self.stable_y),
                    hStack(QLabel("Forward radius:"), self.forward_rad),
                    hStack(QLabel("Backward radius:"), self.backward_rad),
                    hStack(QLabel("Count:"), self.count),
                    hStack(QLabel("Resolution:"), self.resolution_sb, QLabel("**3")),
                    hStack(QLabel("Marker size:"), self.marker_size),
                    self.connect_checker,
                    self.draw_btn
                ),
                vStack(
                    self.equation_cb,
                    self.equation_text
                )
            )
        )
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top_wdg)
        splitter.addWidget(bot_wgd)
        self.setLayout(
            vStack(
                splitter
            )
        )

    def _compute(self):
        self.context, self.queue = createContextAndQueue()

        try:
            self.program = cl.Program(self.context, "\n".join((self.equation_text.toPlainText(), phase_src))).build(
                options=[dummyOption()]
            )
        except cl.RuntimeError as err:
            print("Error:", err.routine)
            print("Error code:", err.code)
            print("Build log:\n", self.program.get_build_info(self.device, cl.program_build_info.LOG))
            print("Build status:\n", self.program.get_build_info(self.device, cl.program_build_info.STATUS))

        resolution = int(self.resolution_sb.value())
        count = int(self.count.value())

        x_f_vec_buffer = cl.Buffer(
            context=self.context,
            flags=cl.mem_flags.WRITE_ONLY,
            size=resolution * count * np.float32().nbytes
        )
        y_f_vec_buffer = cl.Buffer(
            context=self.context,
            flags=cl.mem_flags.WRITE_ONLY,
            size=resolution * count * np.float32().nbytes
        )
        x_b_vec_buffer = cl.Buffer(
            context=self.context,
            flags=cl.mem_flags.WRITE_ONLY,
            size=resolution * count * np.float32().nbytes
        )
        y_b_vec_buffer = cl.Buffer(
            context=self.context,
            flags=cl.mem_flags.WRITE_ONLY,
            size=resolution * count * np.float32().nbytes
        )

        rnd = Random()
        init_f_vec = np.empty((count, 2), dtype=np.float32)
        init_b_vec = np.empty((count, 2), dtype=np.float32)
        for idx in range(count):
            fx = rnd.uniform(
                float(self.stable_x.value()) - float(self.forward_rad.value()),
                float(self.stable_x.value()) + float(self.forward_rad.value())
            )
            fy = rnd.uniform(
                float(self.stable_y.value()) - float(self.forward_rad.value()),
                float(self.stable_y.value()) + float(self.forward_rad.value())
            )
            bx = rnd.uniform(
                float(self.stable_x.value()) - float(self.backward_rad.value()),
                float(self.stable_x.value()) + float(self.backward_rad.value())
            )
            by = rnd.uniform(
                float(self.stable_y.value()) - float(self.backward_rad.value()),
                float(self.stable_y.value()) + float(self.backward_rad.value())
            )
            init_f_vec[idx][0] = fx
            init_f_vec[idx][1] = fy
            init_b_vec[idx][0] = bx
            init_b_vec[idx][1] = by

        init_f_vec_buffer = cl.Buffer(
            context=self.context,
            flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=init_f_vec
        )
        init_b_vec_buffer = cl.Buffer(
            context=self.context,
            flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=init_b_vec
        )

        self.program.homoclinic_plot(
            self.queue, (resolution,), None,

            np.float32(self.a_var.value()),
            np.float32(self.b_var.value()),
            np.int32(self.count.value()),

            init_f_vec_buffer,
            init_b_vec_buffer,

            x_f_vec_buffer,
            y_f_vec_buffer,
            x_b_vec_buffer,
            y_b_vec_buffer
        )

        self.x_f_vec = np.empty(resolution * count, dtype=np.float32)
        self.y_f_vec = np.empty(resolution * count, dtype=np.float32)
        self.x_b_vec = np.empty(resolution * count, dtype=np.float32)
        self.y_b_vec = np.empty(resolution * count, dtype=np.float32)

        cl.enqueue_copy(self.queue, self.x_f_vec, x_f_vec_buffer)
        cl.enqueue_copy(self.queue, self.y_f_vec, y_f_vec_buffer)
        cl.enqueue_copy(self.queue, self.x_b_vec, x_b_vec_buffer)
        cl.enqueue_copy(self.queue, self.y_b_vec, y_b_vec_buffer)

    def _draw(self):
        self._compute()

        resolution = int(self.resolution_sb.value())
        count = int(self.count.value())

        self.plot_f.clear()
        self.plot_b.clear()
        for idx in range(1, resolution):
            self.plot_f.scatter(self.x_f_vec[idx * count: (idx + 1) * count],
                                self.y_f_vec[idx * count: (idx + 1) * count], s=float(self.marker_size.value()))
            self.plot_b.scatter(self.x_b_vec[idx * count: (idx + 1) * count],
                                self.y_b_vec[idx * count: (idx + 1) * count], s=float(self.marker_size.value()))

        self.figure_f.canvas.draw()
        self.figure_b.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWin = MainWindow()
    mainWin.setWindowTitle("0")
    mainWin.showMaximized()
    sys.exit(app.exec_())
