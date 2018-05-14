import numpy as np
import pyopencl as cl
from PyQt4 import Qt, QtGui, QtCore


DEBUG_CL = (False, 0, 1)


def allocate_image(ctx, w, h, flags=cl.mem_flags.WRITE_ONLY) :
    fmt = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNORM_INT8)
    return np.empty((w, h, 4), dtype=np.uint8), cl.Image(ctx, flags, fmt, shape=(w, h))


def create_context_and_queue(config=None):
    if config is None or config.get("autodetect"):
        ctx = cl.create_some_context(interactive=False)
        print("Using auto-detected device:", ctx.get_info(cl.context_info.DEVICES))
    else:
        pl = cl.get_platforms()[config.get("pid", 0)]
        dev = pl.get_devices()[config.get("did", 0)]
        print("Using specified device:", dev)
        ctx = cl.Context([dev])

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

    def clamp_x(self, v):
        return np.clip(v, self.x_min, self.x_max)

    def from_integer_x(self, v, v_max):
        return self.x_min + v / v_max * (self.x_max - self.x_min)

    def clamp_y(self, v):
        return np.clip(v, self.y_min, self.y_max)

    def from_integer_y(self, v, v_max, invert=True):
        return self.y_min + ((v_max - v) if invert else v)/ v_max * (self.y_max - self.y_min)

    def to_integer(self, x, y, w, h, invert_y=True):
        y_val = int((y - self.y_min) / (self.y_max - self.y_min) * h)
        return (
            int((x - self.x_min) / (self.x_max - self.x_min) * w),
            y_val if not invert_y else h - y_val
        )

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
        
        #define convert_real convert_{}
        #define convert_real2 convert_{}2
        #define convert_real3 convert_{}3
        #define convert_real4 convert_{}4
        """.format(*[TYPES[self.real_type], ] * 8)

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


def generate_variable_name(idx):
    return "_variable__%02d" % (idx,)


def generate_var_code(param_count):
    if param_count > 4:
        raise ValueError("Supported dimensonalities are 1-4 (%d requested)" % (param_count,))

    names = [generate_param_name(i) for i in range(param_count)]
    signatures = "#define VARIABLE_SIGNATURES " + ", ".join(["real " + name for name in names])
    values = "#define VARIABLE_VALUES " + ", ".join(names)
    if param_count == 1:
        var_type = "real"
        compare = "#define VARIABLE_VECTOR_NEAR(v1, v2, val) (fabs(v1 - v2) < val)"
        compare_greater = "#define VARIABLE_VECTOR_ANY_ABS_GREATER(v, val) (fabs(v) > val)"
    else:
        var_type = "real" + str(param_count)
        compare = "#define VARIABLE_VECTOR_NEAR(v1, v2, val) \\\n\t(" + "&&".join([
            "(fabs(v1.s%01d - v2.s%01d) < val)" % (i, i) for i in range(param_count)
        ]) + ")"
        compare_greater = "#define VARIABLE_VECTOR_ANY_ABS_GREATER(v, val) \\\n\t(" + "||".join([
            "(fabs(v.s%01d) > val)" % (i,) for i in range(param_count)
        ]) + ")"

    gather = "#define GATHER_VARIABLES (%s)(VARIABLE_VALUES)" % (var_type,)
    acceptor = "#define VARIABLE_ACCEPTOR_TYPE " + var_type
    return "\n".join([signatures, values, gather, compare, compare_greater, acceptor])


def make_param_list(total_params, params, type, active_idx=None):
    if total_params < len(params):
        params = params[:total_params] # todo raise warning?
    if total_params == len(params):
        return list(map(type, params))
    if total_params - 1 == len(params) and active_idx is not None:
        return list(map(type, params[:active_idx])) + [type(0.0),] + list(map(type, params[active_idx+1:]))
    raise ValueError("Out of %d arguments, only %d were provided." % (total_params, len(params)))


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
        self.shape = shape

        self.current_selection = None, None

        def _custom_mouse_move(x, y, lmb, rmb):
            h_val, v_val = self._convert(x, y)
            self.print_current(x, y)
            if lmb and any((h_val, v_val)):
                self.current_selection = h_val, v_val
                self.selectionChanged.emit(h_val or 0, v_val or 0)
            custom_mouse_move(x, y, lmb, rmb)  # call next user-defined function

        self.image_widget = ImageWidget(custom_mouse_move=_custom_mouse_move,
                                        crosshair=Crosshair(crosshair_color, shape=shape))

        self.setLayout(qt_vstack(self.image_widget, self.position_label))

    def print_current(self, x=None, y=None):
        cur = ""
        if x is not None and y is not None and any(self.shape):
            # print x and y:
            h_val, v_val = self._convert(x, y)
            h_comp = ("%s = %f " % (self.x_name, h_val)) if self.shape[0] else ""
            v_comp = ("%s = %f" % (self.y_name, v_val)) if self.shape[1] else ""
            sep = "; " if h_comp and v_comp else ""
            cur = h_comp + sep + v_comp

        sel = ""
        xs, ys = self.current_selection
        if xs is not None and ys is not None and any(self.shape):
            # print selection
            h_selected = ("%s = %f " % (self.x_name, self.current_selection[0])) \
                if self.shape[0] and self.current_selection[0] is not None else ""
            v_selected = ("%s = %f " % (self.y_name, self.current_selection[1])) \
                if self.shape[1] and self.current_selection[1] is not None else ""
            sep = "; " if h_selected and v_selected else ""
            sel = h_selected + sep + v_selected

        if any((cur, sel)):
            sep = "  |  " if all((cur, sel)) else ""
            self.position_label.setText(cur + sep + sel)

    def _convert(self, x, y):
        return self.bounds.clamp_x(self.bounds.from_integer_x(x, self.image_widget.w)) if self.shape[0] else None, \
               self.bounds.clamp_y(self.bounds.from_integer_y(y, self.image_widget.h, invert=True)) if self.shape[1] else None

    def set_image(self, image):
        self.image_widget.set_numpy_image(image)

    def set_crosshair_pos(self, x, y):
        self.image_widget.crosshair.pos(x, y)
        self.current_selection = self.bounds.from_integer_x(x, self.image_widget.w),\
                                 self.bounds.from_integer_y(y, self.image_widget.h, invert=False)
        self.print_current(x, y)
        self.image_widget.repaint()

    def get_selection(self):
        return self.current_selection


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

    def read_from_device(self, queue=None):
        cl.enqueue_copy(queue if queue is not None else self.queue, self.image, self.image_device, origin=(0, 0), region=(self.width, self.height))
        return self.image


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


COMMON_SOURCE = """

#define ID_2D (int2)(get_global_id(0), get_global_id(1))
#define ID_2D_Y_INV (int2)(get_global_id(0), get_global_size(1) - get_global_id(1))

#define SIZE_2D (int2)(get_global_size(0), get_global_size(1))

#define TRANSLATE(id, size, min_, max_) \
    ((min_) + (id)*((max_) - (min_))/(size))

#define TRANSLATE_BACK(v, min_, max_, size) \
    (((v) - (min_)) / ((max_) - (min_)) * (size))

#define TRANSLATE_BACK_INV(v, min_, max_, size) \
    ((size) - TRANSLATE_BACK((v), (min_), (max_), (size)))

#define TRANSLATE_2D(id, size, x_min, x_max, y_min, y_max) \
    (real2)((x_min) + (id).x*((x_max) - (x_min))/(size).x, (y_min) + (id).y*((y_max) - (y_min))/(size).y)

#define TRANSLATE_2D_INV_Y(id, size, x_min, x_max, y_min, y_max) \
    (real2)((x_min) + (id).x*((x_max) - (x_min))/(size).x, (y_min) + ((size).y - (id).y)*((y_max) - (y_min))/((size).y))

#define TRANSLATE_BACK_2D(v, x_min, x_max, y_min, y_max, size) \
    convert_int2_rtz( (real2) (((v).x - (x_min))/((x_max) - (x_min))*(size).x, \
                               ((v).y - (y_min))/((y_max) - (y_min))*(size).y ))

#define TRANSLATE_BACK_2D_INV_Y(v, x_min, x_max, y_min, y_max, size) \
    convert_int2_rtz( (real2) ( ((v).x - (x_min))/((x_max) - (x_min))*(size).x, \
                                (size).y - ((v).y - (y_min))/((y_max) - (y_min))*(size).y ))

#define NEAR(a, b, abs_error) (fabs((a) - (b)) < (abs_error))

float3 hsv2rgb(float3 hsv) {
    const float c = hsv.y * hsv.z;
    const float x = c * (1 - fabs(fmod( hsv.x / 60, 2 ) - 1));
    float3 rgb;
    if      (0 <= hsv.x && hsv.x < 60) {
        rgb = (float3)(c, x, 0);
    } else if (60 <= hsv.x && hsv.x < 120) {
        rgb = (float3)(x, c, 0);
    } else if (120 <= hsv.x && hsv.x < 180) {
        rgb = (float3)(0, c, x);
    } else if (180 <= hsv.x && hsv.x < 240) {
        rgb = (float3)(0, x, c);
    } else if (240 <= hsv.x && hsv.x < 300) {
        rgb = (float3)(x, 0, c);
    } else {
        rgb = (float3)(c, 0, x);
    }
    return (rgb + (hsv.z - c));
}


"""


def make_cl_source(*args, type_config=None):
    return type_config.cl() + "\n" + COMMON_SOURCE + "\n" + "\n".join(args)