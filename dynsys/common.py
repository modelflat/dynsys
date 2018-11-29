import numpy as np
import pyopencl as cl
from PyQt5 import Qt, QtCore

from .ui import vStack as qt_vstack
from .ui.image_widgets import toPixmap as to_pixmap



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


class ImageWidget(Qt.QLabel):

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
        self.current_selection = self.bounds.from_integer_x(x, self.image_widget.w), \
                                 self.bounds.from_integer_y(y, self.image_widget.h, invert=False)
        self.print_current(x, y)
        self.image_widget.repaint()

    def get_selection(self):
        return self.current_selection


class ComputedImage:

    def __init__(self, ctx, queue, width, height, bounds, *sources, type_config):
        self.ctx, self.queue, self.tc = ctx, queue, type_config
        self.width, self.height = width, height
        self.image, self.image_device = allocate_image(ctx, width, height)
        self.bounds = bounds
        self.program = cl.Program(ctx, make_cl_source(
            *sources, type_config=type_config
        )).build()

    def clear(self, read_back=False, color=(1.0, 1.0, 1.0, 1.0)):
        cl.enqueue_fill_image(self.queue, self.image_device, np.array(color), origin=(0, 0), region=(self.width, self.height))
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


# COMMON_SOURCE = """
#
# #define ID_2D (int2)(get_global_id(0), get_global_id(1))
# #define ID_2D_Y_INV (int2)(get_global_id(0), get_global_size(1) - get_global_id(1))
#
# #define SIZE_2D (int2)(get_global_size(0), get_global_size(1))
#
# #define TRANSLATE(id, size, min_, max_) \
#     ((min_) + (id)*((max_) - (min_))/(size))
#
# #define TRANSLATE_BACK(v, min_, max_, size) \
#     (((v) - (min_)) / ((max_) - (min_)) * (size))
#
# #define TRANSLATE_BACK_INV(v, min_, max_, size) \
#     ((size) - TRANSLATE_BACK((v), (min_), (max_), (size)))
#
# #define TRANSLATE_2D(id, size, x_min, x_max, y_min, y_max) \
#     (real2)((x_min) + (id).x*((x_max) - (x_min))/(size).x, (y_min) + (id).y*((y_max) - (y_min))/(size).y)
#
# #define TRANSLATE_2D_INV_Y(id, size, x_min, x_max, y_min, y_max) \
#     (real2)((x_min) + (id).x*((x_max) - (x_min))/(size).x, (y_min) + ((size).y - (id).y)*((y_max) - (y_min))/((size).y))
#
# #define TRANSLATE_BACK_2D(v, x_min, x_max, y_min, y_max, size) \
#     convert_int2_rtz( (real2) (((v).x - (x_min))/((x_max) - (x_min))*(size).x, \
#                                ((v).y - (y_min))/((y_max) - (y_min))*(size).y ))
#
# #define TRANSLATE_BACK_2D_INV_Y(v, x_min, x_max, y_min, y_max, size) \
#     convert_int2_rtz( (real2) ( ((v).x - (x_min))/((x_max) - (x_min))*(size).x, \
#                                 (size).y - ((v).y - (y_min))/((y_max) - (y_min))*(size).y ))
#
# #define NEAR(a, b, abs_error) (fabs((a) - (b)) < (abs_error))
#
# float3 hsv2rgb(float3);
#
# float3 hsv2rgb(float3 hsv) {
#     const float c = hsv.y * hsv.z;
#     const float x = c * (1 - fabs(fmod( hsv.x / 60, 2 ) - 1));
#     float3 rgb;
#     if      (0 <= hsv.x && hsv.x < 60) {
#         rgb = (float3)(c, x, 0);
#     } else if (60 <= hsv.x && hsv.x < 120) {
#         rgb = (float3)(x, c, 0);
#     } else if (120 <= hsv.x && hsv.x < 180) {
#         rgb = (float3)(0, c, x);
#     } else if (180 <= hsv.x && hsv.x < 240) {
#         rgb = (float3)(0, x, c);
#     } else if (240 <= hsv.x && hsv.x < 300) {
#         rgb = (float3)(x, 0, c);
#     } else {
#         rgb = (float3)(c, 0, x);
#     }
#     return (rgb + (hsv.z - c));
# }
#
#
# """
#
#
# def make_cl_source(*args, type_config=None):
#     return type_config.cl() + "\n" + COMMON_SOURCE + "\n" + "\n".join(args)