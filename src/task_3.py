from PyQt4 import QtGui, Qt, QtCore
import pyopencl as cl
import numpy as np
import sys

import kernels3


def create_context():
    c = cl.create_some_context(answers=[1, 0])
    print(c.get_info(cl.context_info.DEVICES))
    return c


def allocate_image(ctx, w, h) :
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    return cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))


def pixmap_from_raw_image(img: np.ndarray):
    image = QtGui.QImage(img.data, img.shape[0], img.shape[1], QtGui.QImage.Format_ARGB32)
    pixmap = QtGui.QPixmap()
    pixmap.convertFromImage(image)
    return pixmap, image


def type():
    return np.float64

def real(arg):
    return type()(arg)

def typesize():
    return 8


def translate(x, d, a, b):
    return a + (b - a)*x / d


class BifurcationTree:

    def __init__(self, ctx, queue, w, h):
        self.ctx, self.queue, self.w, self.h = ctx, queue, w, h
        self.program = cl.Program(ctx, kernels3.BIFURCATION_TREE_KERNEL_SOURCE).build()
        self.image_device = allocate_image(ctx, w, h)
        self.image = np.empty((w, h, 4), dtype=np.uint8)

    def compute(self, A, B, start, stop, x_start, skip, samples_count, x_max):
        self.result_device = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=samples_count * self.w * typesize())
        self.result_minmax_device = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=self.w * 2 * typesize())

        self.program.prepare_bifurcation_tree(
            self.queue, (self.w, ), None,
            real(np.nan if A is None else A), real(np.nan if B is None else B),
            np.int32(1 if A is not None else 0),
            real(start), real(stop),
            real(x_start), real(x_max),
            np.int32(skip), np.int32(samples_count),
            self.result_device, self.result_minmax_device
        )
        result_minmax = np.empty((self.w*2,), dtype=type())

        cl.enqueue_copy(self.queue, result_minmax, self.result_minmax_device)

        return self.result_device, min(result_minmax), max(result_minmax)

    def draw(self, A, B, start, stop, x_start, skip, samples_count=None):
        self.result_device, min_, max_ = self.compute(A, B, start, stop, x_start, skip,
                                                 self.h if samples_count is None else samples_count, 500)
        self.clear()

        # print(min_, max_)
        # print("drawing")

        self.program.draw_bifurcation_tree(
            self.queue, (self.w, ), None,
            self.result_device,
            np.int32(samples_count),
            real(min_), real(max_),
            real(self.h),
            self.image_device
        )

        # print("drawn")
        # print( self.image.shape, self.image.size, self.image_device.__sizeof__())

        cl.enqueue_copy(self.queue, self.image, self.image_device, origin=(0, 0), region=(self.w, self.h))
        # print("copy-out")
        return pixmap_from_raw_image(self.image)

    def clear(self, read_back=False):
        self.program.clear(self.queue, (self.w, self.h), None, self.image_device)
        if read_back:
            cl.enqueue_copy(self.queue, self.image, self.image_device, origin=(0, 0), region=(self.w, self.h))



def compute_dynamic_map(ctx, prg, queue, w, h, x_min, y_min, x_max, y_max, start, samples_count, skip, image=None):
    if image is None:
        result_device = allocate_image(ctx, w, h)
    else:
        result_device = image
    global samples_device
    samples_device = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=w*h*samples_count*typesize())

    prg.compute_map(queue, (w, h), None,
                    real(x_min), real(x_max), real(y_min), real(y_max),
                    real(start), np.int32(samples_count), np.int32(skip),
                    samples_device, result_device )

    result = np.empty((w, h, 4), dtype=np.uint8)

    cl.enqueue_copy(queue, result, result_device, origin=(0, 0), region=(w, h))
    return result


class ImageWidget(QtGui.QLabel):

    def onMouseMove(self, fn, when_pressed):
        self._onMouseMove = fn
        self._onMouseMoveWhenPressed = when_pressed

    def mouseMoveEvent(self, QMouseEvent):
        if self._onMouseMove:
            self._onMouseMove(QMouseEvent.x(), QMouseEvent.y(), QMouseEvent)
            if hasattr(self, "redraw") and self.redraw:
                self._onMouseMoveWhenPressed(QMouseEvent.x(), QMouseEvent.y(), QMouseEvent)


    def onMouseClick(self, fn):
        self._onMouseClick = fn

    def mouseReleaseEvent(self, QMouseEvent):
        if self._onMouseClick:
            self._onMouseClick(QMouseEvent.x(), QMouseEvent.y(), QMouseEvent)
        self.redraw = False

    def mousePressEvent(self, QMouseEvent):
        self.redraw = True

    def onPaint(self, fn):
        self._customPaintEvent = fn

    def paintEvent(self, QPaintEvent):
        super().paintEvent(QPaintEvent)
        if hasattr(self, "_customPaintEvent"):
            self._customPaintEvent(self, QPaintEvent)


class App(QtGui.QWidget):
    def __init__(self, parent=None):
        self.w, self.h = 512, 512
        self.ctx = create_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernels3.DYNAMIC_MAP_KERNEL_SOURCE).build()

        self.bif_tree = BifurcationTree(self.ctx, self.queue, self.w, self.h)

        super(App, self).__init__(parent)

        self.setWindowTitle('Dynamic Map')

        self.x_min = -1.2
        self.x_max = 1.2
        self.y_min = 0
        self.y_max = 2.5

        self.coord_label = QtGui.QLabel()
        self.image_label = ImageWidget()
        self.image_label.setMouseTracking(True)
        self.image2_label = ImageWidget()
        self.image2_label.setMouseTracking(True)

        def draw_tree_lam(x, y, ev):
            self.pA, self.pA_d = translate(x, self.w, self.x_min, self.x_max), x
            self.pB, self.pB_d = translate(self.h - y, self.h, self.y_min, self.y_max), y
            self.draw_tree(self.pA, self.pB)
            self.image_label.repaint()

        self.pA, self.pB = None, None
        self.pA_d, self.pB_d = -1, -1

        self.image_label.onMouseMove(lambda x, y, ev: self.coord_label.setText("a = %f; b = %f" % (
            translate(x, self.w, self.x_min, self.x_max),
            translate(self.h - y, self.h, self.y_min, self.y_max)
        )), draw_tree_lam)
        self.image_label.onMouseClick(draw_tree_lam)

        def custom_paintEvent_crosshair(self2, evt):
            painter = Qt.QPainter(self2)
            pen = Qt.QPen(QtCore.Qt.white, 1)
            painter.setPen(pen)
            painter.drawLine(0, self.pB_d, self.w, self.pB_d)
            painter.drawLine(self.pA_d, 0, self.pA_d, self.h)

        self.image_label.onPaint(custom_paintEvent_crosshair)

        self.image2_label.onMouseMove(lambda x, y, ev: not any(self.last_params) or self.coord_label.setText("a = %f, b = %f" % (
            translate(x, self.w, self.x_min, self.x_max) if not self.use_param_A else self.last_params[0],
            translate(x, self.w, self.x_min, self.x_max) if self.use_param_A else self.last_params[1]
        )), lambda *args: None)

        def custom_paintEvent(self2, evt):
            if not any(self.last_params):
                return
            painter = Qt.QPainter(self2)
            pen = Qt.QPen(QtCore.Qt.red, 1)
            painter.setPen(pen)
            v = (abs(self.active_param_value()) ) / (self.x_max ) * self.w
            painter.drawLine(v, 0, v, self.h)

        self.image2_label.onPaint(custom_paintEvent)

        layout = QtGui.QVBoxLayout()
        layout_images = QtGui.QHBoxLayout()

        layout_images.addWidget(self.image_label)
        layout_images.addWidget(self.image2_label)

        layout.addLayout(layout_images)

        layout.addWidget(self.coord_label)
        self.slider = QtGui.QSlider()
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setMinimum(-100)
        self.slider.setMaximum(100)
        self.slider.setSingleStep(.01)
        self.slider.sliderMoved.connect(lambda val: (self.draw_map(val),))
        layout.addWidget(self.slider)
        self.slider.setDisabled(True)
        self.setLayout(layout)
        self.start = 1.0
        self.mul = .005
        self.image = allocate_image(self.ctx, self.w, self.h )
        self.image_bif = allocate_image(self.ctx, self.w, self.h)

        self.use_param_A = False
        self.skip = 512
        self.samples = self.w
        self.samples_map = 100

        self.draw_map(0.0)
        self.last_params = (None, None)

    def create_map_image_widget(self):
        pass

    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.key() == QtCore.Qt.Key_Q:
            print("switching param is disabled")
            # self.use_param_A = not self.use_param_A
        if QKeyEvent.key() == QtCore.Qt.Key_I and all(self.last_params):
            pass

    def mouseDoubleClickEvent(self, event):
        self.draw_map(self.start)

    def draw_map(self, start):
        img = compute_dynamic_map(self.ctx, self.prg, self.queue, self.w, self.h,
                                  self.x_min, self.y_min, self.x_max, self.y_max,
                                   start*self.mul, samples_count=self.samples_map, skip=self.samples+self.skip-self.samples_map+1,
                                  image=self.image)
        self.image_label.setPixmap(pixmap_from_raw_image(img)[0])

    def draw_tree(self, param_A, param_B):
        # print("Drawing for a = %f, b = %f" % (param_A, param_B))
        self.last_params = param_A, param_B
        param_A = None if not self.use_param_A else param_A
        param_B = None if self.use_param_A else param_B
        if self.use_param_A:
            self._p, self._i = self.bif_tree.draw(param_A, param_B, self.y_min, self.y_max, self.start, self.skip, self.samples)
        else:
            self._p, self._i = self.bif_tree.draw(param_A, param_B, 0, self.x_max, self.start, self.skip, self.samples)

        # print("TOTALLY DONE")
        self.image2_label.setPixmap(self._p)

    def active_param_value(self):
        return self.last_params[0 if not self.use_param_A else 1]

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
