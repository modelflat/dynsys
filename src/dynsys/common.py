import numpy as np
import pyopencl as cl
from .common_source import COMMON_SOURCE
from PyQt4 import Qt, QtGui, QtCore
import sys


def make_cl_source(*args, type_config=None):
    return type_config.cl() + "\n" + COMMON_SOURCE + "\n" + "\n".join(args)


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

    def from_integer_x(self, v, v_max):
        return self.x_min + v / v_max * (self.x_max - self.x_min)

    def clamp_x(self, v):
        return np.clip(v, self.x_min, self.x_max)

    def clamp_y(self, v):
        return np.clip(v, self.y_min, self.y_max)

    def from_integer_y(self, v, v_max, invert=True):
        return self.y_min + ((v_max - v) if invert else v)/ v_max * (self.y_max - self.y_min)

    @staticmethod
    def x(x_min, x_max):
        return Bounds(x_min, x_max, None, None)

    @staticmethod
    def y(y_min, y_max):
        return Bounds(None, None, y_min, y_max)


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


def generate_param_name(idx):
    return "_parameter__%02d" % (idx,)


def generate_param_code(param_count):
    names = [generate_param_name(i) for i in range(param_count)]
    signatures = "#define PARAM_SIGNATURES " + ", ".join(["real " + name for name in names])
    values = "#define PARAM_VALUES " + ", ".join(names)
    set_param = "#define SET_PARAM_VALUE(idx, value) {\\\n\t" + "; \\\n\t".join([
        "if ((idx) == %d) %s = (value)" % (i, name) for i, name in enumerate(names)
    ]) + "; }"
    return "\n".join([signatures, values, set_param])


def make_param_list(total_params, params, type, active_idx=None):
    if total_params < len(params):
        params = params[:total_params] # todo raise warning?
    if total_params == len(params):
        return list(map(type, params))
    if total_params - 1 == len(params) and active_idx is not None:
        return list(map(type, params[:active_idx])) + [type(0.0),] + list(map(type, params[active_idx+1:]))
    raise ValueError("Out of %d arguments, only %d were provided." % (total_params, len(params)))


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

    def __init__(self, color=QtCore.Qt.red, shape=(True, True)):
        self.x, self.y = -1, -1
        self.color = color
        self.shape = shape

    def pos(self, x, y):
        self.x, self.y = x, y

    def draw(self, w, h, painter: Qt.QPainter):
        pen = Qt.QPen(self.color, 1)
        painter.setPen(pen)
        if self.shape[1]:
            painter.drawLine(0, self.y, w, self.y)
        if self.shape[0]:
            painter.drawLine(self.x, 0, self.x, h)


class ImageWidget(QtGui.QLabel):

    def __init__(self, crosshair=Crosshair(), custom_paint=lambda *args: None, custom_mouse_move=lambda *args: None):
        super(ImageWidget, self).__init__()
        self.crosshair = crosshair
        self.setMouseTracking(True)
        self.custom_paint = custom_paint
        self.custom_mouse_move = custom_mouse_move
        self.image = None
        self.w, self.h = 1, 1

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

    selectionChanged = Qt.pyqtSignal(float, float)

    def __init__(self, bounds, names=("x", "y"), shape=(True, True), crosshair_color=QtCore.Qt.red, custom_mouse_move=lambda *args:None, parent=None):
        super(ParametrizedImageWidget, self).__init__(parent)
        self.bounds = bounds
        self.x_name, self.y_name = names

        self.position_label = Qt.QLabel()

        def _custom_mouse_move(x, y, lmb, rmb):
            h_val = self.bounds.clamp_x(
                self.bounds.from_integer_x(x, self.image_widget.w)
            ) if shape[0] else 0.0
            h_comp = ("%s = %f " % (self.x_name, h_val)) if shape[0] else ""
            v_val = self.bounds.clamp_y(
                self.bounds.from_integer_y(y, self.image_widget.h, invert=True)
            ) if shape[1] else 0.0
            v_comp = ("%s = %f" % (self.y_name, v_val)) if shape[1] else ""
            if any(shape):
                self.position_label.setText("%s%s | [ %s ]" % (h_comp, v_comp, "selecting" if lmb else "looking"))
            if lmb:
                self.selectionChanged.emit(h_val, v_val)
            custom_mouse_move(x, y, lmb, rmb)  # call next user-defined function

        self.image_widget = ImageWidget(custom_mouse_move=_custom_mouse_move,
                                        crosshair=Crosshair(crosshair_color, shape=shape))

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
        super().valueChanged.connect(self._value_changed)

    @QtCore.pyqtSlot(int)
    def _value_changed(self, int_val):
        self.valueChanged.emit(self.value())

    def set_value(self, v):
        super().setValue( int( (v - self.min_val) / (self.max_val - self.min_val) * self.steps ) )

    def value(self):
        return float(super().value()) / self.steps * (self.max_val - self.min_val) + self.min_val

    @staticmethod
    def makeAndConnect(min_val, max_val, current_val=None, steps=10000,
                       horizontal=True, connect_to=None):
        s = RealSlider(min_val, max_val, steps=steps, horizontal=horizontal)
        if connect_to is not None:
            s.valueChanged.connect(connect_to)
        s.set_value(current_val if current_val is not None else min_val)
        return s


class IntegerSlider(QtGui.QSlider):

    def __init__(self, min_val, max_val, horizontal=True):
        QtGui.QSlider.__init__(self)
        super().setOrientation(QtCore.Qt.Vertical if not horizontal else QtCore.Qt.Horizontal)
        super().setMinimum(min_val)
        super().setMaximum(max_val)

    @staticmethod
    def makeAndConnect(min_val, max_val, current_val=None, horizontal=True, connect_to=None):
        s = IntegerSlider(min_val, max_val, horizontal=horizontal)
        if connect_to is not None:
            s.valueChanged.connect(connect_to)
        s.setValue(current_val if current_val is not None else min_val)
        return s


class ComputedImage:

    def __init__(self, ctx, queue, width, height, bounds, *sources, type_config=float_config):
        self.ctx, self.queue, self.tc = ctx, queue, type_config
        self.width, self.height = width, height
        self.image, self.image_device = allocate_image(ctx, width, height)
        self.bounds = bounds
        self.program = cl.Program(ctx, make_cl_source(
            *sources, type_config=type_config
        )).build()

    def clear(self, read_back=False, color=(1.0, 1.0, 1.0, 1.0)):
        cl.enqueue_fill_image(self.queue, self.image_device, color, origin=(0, 0), region=(self.width, self.height))
        if read_back:
            cl.enqueue_copy(self.queue, self.image, self.image_device, origin=(0, 0), region=(self.width, self.height))


class ObservableValue(Qt.QObject):

    valueChanged = Qt.pyqtSignal(object)

    def __init__(self, initial):
        super().__init__()
        self._value = initial

    def value(self):
        return self._value

    def setValue(self, v):
        self._value = v
        self.valueChanged.emit( v )

    @staticmethod
    def makeAndConnect(initial, connect_to=None):
        o = ObservableValue(initial)
        if connect_to is not None:
            o.valueChanged.connect(connect_to)
        return o
