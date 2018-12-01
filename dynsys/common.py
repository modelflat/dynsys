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

