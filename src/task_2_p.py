from PyQt4 import QtGui, Qt, QtCore
import pyopencl as cl
import numpy as np
import sys

BIFURCATION_TREE_KERNEL_SOURCE = """

#define real double
#define real2 double2

real map_function(real x, real param_A) {
    return x * (param_A - x*x);
}

kernel void clear(write_only image2d_t img) {
    write_imageui(img, (int2)(get_global_id(0), get_global_id(1)), (uint4)(0));
}

kernel void prepare_bifurcation_tree(
    const real start, const real stop,
    const real x_start, const real x_max,
    const int skip, const int samples_count,

    global real* result,
    global real2* result_minmax
) {
    const int id = get_global_id(0);
    const real value = start + ( (stop - start)*id ) / get_global_size(0);

    real x = x_start;
    real min_ = x_start;
    real max_ = x_start;
    for (int i = 0; i < skip; ++i) {
        x = map_function(x, value);
    }

    for (int i = 0; i < samples_count; ++i) {
        x = map_function(x, value);
        if (x < min_ && x > -x_max ) min_ = x;
        if (x > max_ && x < x_max) max_ = x;
        result[id * samples_count + i] = x;
    }

    result_minmax[id] = (real2)(min_, max_); // save minmax
}

kernel void draw_bifurcation_tree(
    const global double* samples,
    const int samples_count,
    const real min_, const real max_,
    const real height,
    write_only image2d_t result
) {
    const int id = get_global_id(0);
    const real part = (max_ - min_) / height;
    samples += id * samples_count;
    for (int i = 0; i < samples_count; ++i) {
        int y_coord = (samples[i] - min_) / part;
        write_imageui(result, (int2)(id, height - y_coord ), (uint4)((uint3)(0), 255));
    }
}

"""


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
        self.program = cl.Program(ctx, BIFURCATION_TREE_KERNEL_SOURCE).build()
        self.image_device = allocate_image(ctx, w, h)
        self.image = np.empty((w, h, 4), dtype=np.uint8)

    def compute(self, start, stop, x_start, skip, samples_count, x_max):
        self.result_device = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=samples_count * self.w * typesize())
        self.result_minmax_device = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=self.w * 2 * typesize())

        self.program.prepare_bifurcation_tree(
            self.queue, (self.w, ), None,
            real(start), real(stop),
            real(x_start), real(x_max),
            np.int32(skip), np.int32(samples_count),
            self.result_device, self.result_minmax_device
        )
        result_minmax = np.empty((self.w*2,), dtype=type())

        cl.enqueue_copy(self.queue, result_minmax, self.result_minmax_device)

        return self.result_device, min(result_minmax), max(result_minmax)

    def draw(self, start, stop, x_start, skip, samples_count=None):
        self.result_device, min_, max_ = self.compute(start, stop, x_start, skip,
                                self.h if samples_count is None else samples_count, 500)
        self.clear()

        self.program.draw_bifurcation_tree(
            self.queue, (self.w, ), None,
            self.result_device,
            np.int32(samples_count),
            real(min_), real(max_),
            real(self.h),
            self.image_device
        )

        print("drawn")

        cl.enqueue_copy(self.queue, self.image, self.image_device, origin=(0, 0), region=(self.w, self.h))
        return pixmap_from_raw_image(self.image)

    def clear(self, read_back=False):
        self.program.clear(self.queue, (self.w, self.h), None, self.image_device)
        if read_back:
            cl.enqueue_copy(self.queue, self.image, self.image_device, origin=(0, 0), region=(self.w, self.h))


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

    def onPaint(self, fn):
        self._customPaintEvent = fn

    def paintEvent(self, QPaintEvent):
        super().paintEvent(QPaintEvent)
        if hasattr(self, "_customPaintEvent"):
            self._customPaintEvent(self, QPaintEvent)


class App(QtGui.QWidget):

    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        self.setWindowTitle('Task 2')

        self.w, self.h = 512, 512
        self.ctx = create_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.bif_tree = BifurcationTree(self.ctx, self.queue, self.w, self.h)

        self.coord_label = QtGui.QLabel()

        self.image_label = ImageWidget()
        self.image_label.setMouseTracking(True)
        self.image_label.onMouseMove(lambda x, y, ev: (self.coord_label.setText("lambda = %f" % (
            translate(x, self.w, self.f, self.t),
        ))), lambda x, y, a: self.setPos(x, y))
        self.image_label.onMouseClick(lambda x, y, ev: self.draw_iter())

        def custom_paintEvent(self2, evt):
            if self.last_param is None: return
            painter = Qt.QPainter(self2)
            pen = Qt.QPen(QtCore.Qt.red, 1)
            painter.setPen(pen)
            painter.drawLine(self.pos_x, 0, self.pos_x, self.h)

        self.image_label.onPaint(custom_paintEvent)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.coord_label)

        self.setLayout(layout)

        self.image = allocate_image(self.ctx, self.w, self.h )
        self.image_bif = allocate_image(self.ctx, self.w, self.h)

        self.skip = 512
        self.samples = self.w

        self.start = 1
        self.f, self.t = 1.8, 2.5
        self.last_param = None

        self.draw_tree()

    def setPos(self, x, y):
        self.x_pos = x
        self.image_label.repaint()

    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.key() == QtCore.Qt.Key_I and all(self.last_params):
            print("draw iter diag in pyplot")
            self.draw_iter()

    def draw_iter(self):
        lam = translate(self.x_pos, self.w, self.f, self.t)
        print("drawing for lam = ", lam)

    def draw_tree(self):
        self._p, self._i = self.bif_tree.draw(self.f, self.t, self.start, self.skip, self.samples)
        self.image_label.setPixmap(self._p)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
