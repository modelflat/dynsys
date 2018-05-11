import numpy as np
import pyopencl as cl
from .common_source import COMMON_SOURCE
from PyQt4 import Qt, QtGui, QtCore
import sys


def make_cl_source(*args, type_config=None):
    return type_config.cl() + "\n" + \
           COMMON_SOURCE + "\n" + "\n".join(args)


def allocate_image(ctx, w, h, flags=cl.mem_flags.WRITE_ONLY) :
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    return np.empty((w, h, 4), dtype=np.uint8), cl.Image(ctx, flags, fmt, shape=(w, h))


def create_context_and_queue():
    ctx = cl.create_some_context()
    return ctx, cl.CommandQueue(ctx)


def to_pixmap(img):
    image = QtGui.QImage(img.data, img.shape[0], img.shape[1], QtGui.QImage.Format_ARGB32)
    pixmap = QtGui.QPixmap()
    pixmap.convertFromImage(image)
    return pixmap


def qt_vstack(*args):
    l = QtGui.QVBoxLayout()
    for a in args:
        if isinstance(a, Qt.QLayout):
            l.addLayout(a)
        else:
            l.addWidget(a)
    return l


def qt_hstack(*args):
    l = QtGui.QHBoxLayout()
    for a in args:
        if isinstance(a, Qt.QLayout):
            l.addLayout(a)
        else:
            l.addWidget(a)
    return l



class Bounds:

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def from_integer(self, v, v_max):
        return self.x_min + v / v_max * (self.x_max - self.x_min)


TYPES = {
    np.float16: "half",
    np.float32: "float",
    np.float64: "double"
}


class TypeConfig:
    def __init__(self, real_type):
        self.real_type = real_type

    def real(self, arg=None):
        return self.real_type(arg) if arg is not None else self.real_type

    def real_size(self):
        return np.dtype(self.real_type).itemsize

    def cl(self):
        return """
        #define real {}
        #define real2 {}2
        #define real3 {}3
        #define real4 {}4
        """.format(*[TYPES[self.real_type], ] * 4)

    def __call__(self):
        return self.real_type, self.real_size()


float_config = TypeConfig(np.float32)
double_config = TypeConfig(np.float64)


class SimpleApp(QtGui.QWidget):

    def __init__(self, title):
        self.app = QtGui.QApplication(sys.argv)
        super(SimpleApp, self).__init__(None)
        self.setWindowTitle(title)
        self.ctx, self.queue = create_context_and_queue()

    def run(self):
        self.show()
        sys.exit(self.app.exec_())


class Crosshair:

    def __init__(self, color=QtCore.Qt.red):
        self.x, self.y = -1, -1
        self.color = color

    def pos(self, x, y):
        self.x, self.y = x, y

    def draw(self, w, h, painter: Qt.QPainter):
        pen = Qt.QPen(self.color, 1)
        painter.setPen(pen)
        painter.drawLine(0, self.y, w, self.y)
        painter.drawLine(self.x, 0, self.x, h)


class ImageWidget(QtGui.QLabel):

    def __init__(self, crosshair=Crosshair(), custom_paint=lambda *args: None, custom_mouse_move=lambda *args: None):
        super(ImageWidget, self).__init__()
        self.crosshair = crosshair
        self.setMouseTracking(True)
        self.custom_paint = custom_paint
        self.custom_mouse_move = custom_mouse_move
        self.image = None

    def mouseMoveEvent(self, QMouseEvent):
        super(ImageWidget, self).mouseMoveEvent(QMouseEvent)
        lmb_pressed = QtCore.Qt.LeftButton & QMouseEvent.buttons()
        rmb_pressed = QtCore.Qt.RightButton & QMouseEvent.buttons()
        self.custom_mouse_move(QMouseEvent.x(), QMouseEvent.y(), lmb_pressed, rmb_pressed)
        if lmb_pressed:
            self.crosshair.pos(QMouseEvent.x(), QMouseEvent.y())
            self.repaint()

    def paintEvent(self, QPaintEvent):
        super(ImageWidget, self).paintEvent(QPaintEvent)
        self.crosshair.draw(self.width(), self.height(), Qt.QPainter(self))

    def set_numpy_image(self, image):
        self.image = image
        self.w, self.h = image.shape[:2]
        self.setPixmap(to_pixmap(image))


class ParametrizedImageWidget(Qt.QWidget):

    def __init__(self, bounds, names, crosshair_color=QtCore.Qt.red, custom_mouse_move=lambda *args:None, parent=None):
        super(ParametrizedImageWidget, self).__init__(parent)
        self.bounds = bounds
        self.x_name, self.y_name = names

        self.position_label = Qt.QLabel()

        def _custom_mouse_move(x, y, lmb, rmb):
            self.position_label.setText("%s = %f; %s = %f | [ %s ]" % (
                self.x_name, self.bounds.from_integer(x, self.image_widget.w),
                self.y_name, self.bounds.from_integer(self.image_widget.h - y, self.image_widget.h),
                "selecting" if lmb else "looking"
            ))
            custom_mouse_move(x, y, lmb, rmb)  # call next user-defined function

        self.image_widget = ImageWidget(custom_mouse_move=_custom_mouse_move, crosshair=Crosshair(crosshair_color))

        self.setLayout(qt_vstack(self.image_widget, self.position_label))

    def set_image(self, image):
        self.image_widget.set_numpy_image(image)


class RealSlider(QtGui.QSlider):

    valueChanged = Qt.pyqtSignal(float)

    def __init__(self, min_val, max_val, steps=10000, horizontal=True):
        QtGui.QSlider.__init__(self)
        self.steps = steps
        self.min_val = min_val
        self.max_val = max_val
        super().setOrientation(QtCore.Qt.Vertical if not horizontal else QtCore.Qt.Horizontal)
        super().setMinimum(0)
        super().setMaximum(self.steps)
        super().valueChanged.connect(self._valueChanged)

    @QtCore.pyqtSlot(int)
    def _valueChanged(self, int_val):
        self.valueChanged.emit(self.value())

    def value(self):
        return float(super().value()) / self.steps * (self.max_val - self.min_val) + self.min_val


class IntegerSlider(QtGui.QSlider):

    def __init__(self, min_val, max_val, horizontal=True):
        QtGui.QSlider.__init__(self)
        super().setOrientation(QtCore.Qt.Vertical if not horizontal else QtCore.Qt.Horizontal)
        super().setMinimum(min_val)
        super().setMaximum(max_val)
