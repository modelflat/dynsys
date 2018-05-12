from PyQt4 import QtGui, Qt, QtCore
import pyopencl as cl
import numpy as np
import sys
import matplotlib.pyplot as pp


def create_context():
    c = cl.create_some_context(answers=[0, 0])
    print(c.get_info(cl.context_info.DEVICES))
    return c


class Map:

    def __init__(self, lam):
        self.lam = lam

    def __call__(self, x, y):
        return (self.lam - x**2)*x

    def plot_curves(self, style1, style2):
        x = np.linspace(-self.lam, self.lam, 100)
        pp.plot(x, x, style1, x, self(x), style2)

    def compute_iterations(self, x, iter_count):
        result = np.ndarray((iter_count, 2), dtype=np.float64)
        for i in range(0, iter_count, 2):
            x_next = self(x)
            result[i] = x, x_next
            result[i + 1] = x_next, x_next
            x = x_next
        return result

    def plot_iterations(self, x0, iter_count, style):
        xl, yl = self.compute_iterations(x0, iter_count).T
        pp.plot(xl, yl, style)

    def plot(self, x0, iter_count, style="r-", style1="g-", style2="b-"):
        self.plot_curves(style1, style2)
        self.plot_iterations(x0, iter_count, style)
        pp.xlabel('Xn'); pp.ylabel('Xn+1')
        pp.grid()

from utils import pixmap_from_raw_image, translate

def allocate_image(ctx, w, h) :
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    return cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))


def real_type():
    return np.float64

def real(arg):
    return real_type()(arg)

def real_type_size():
    return 8


PHASE_PORTRAIT_KERNEL_SOURCE = """

#define real double
#define real2 double2

#define STEP 4e-4

real2 system(real2 v, real m, real b) {
    real2 p = (real2)(
        (1 + b*v.x - v.x*v.x)*v.x - v.x*v.y,
        v.y*(v.x - m)
    );
    return v + STEP*p;
}

kernel void clear(write_only image2d_t img) {
    write_imageui(img, (int2)(get_global_id(0), get_global_id(1)), (uint4)(1.0));
}

kernel void draw_phase_portrait(
    const real a, const real b,
    const real x_min, const real x_max, const real y_min, const real y_max,
    const int step_count, 
    const int w, const int h,
    write_only image2d_t result
) {
    const int2 id = (int2)(get_global_id(0), get_global_id(1));
    const real2 grid_step = (real2)((x_max - x_min)/get_global_size(0), (y_max - y_min)/get_global_size(1));
    
    real2 point = (real2)(x_min + id.x*grid_step.x, y_min + id.y*grid_step.y);
    
    for (int i = 0; i < step_count; ++i) {
        point = system(point, a, b);
        if (step_count - i < 15) {
            int2 coord = (int2)( (point.x - x_min)/(x_max - x_min)*w, (point.y - y_min)/(y_max - y_min)*h);
            write_imageui(result, coord, (uint4)((uint3)(0), 255));
        }
    }
}

"""

PARAMETER_MAP_KERNEL_SOURCE = """
#define real double
#define real2 double2

#define STEP 1e-4

real2 system(real2 v, real m, real b) {
    real2 p = (real2)(
        (1 + b*v.x - v.x*v.x)*v.x - v.x*v.y,
        v.y*(v.x - m)
    );
    return v + STEP*p;
}


float3 color_for_count(int count, int total) {
    if (count == total) {
        return 0.3;
    }
    const float d = count < 8? 1.0 : .5;
    switch(count % 8) {
        case 1:
            return (float3)(1.0, 0.0, 0.0)*d;
        case 2:
            return (float3)(0.0, 1.0, 0.0)*d;
        case 3:
            return (float3)(0.0, 0.0, 1.0)*d;
        case 4:
            return (float3)(1.0, 0.0, 1.0)*d;
        case 5:
            return (float3)(1.0, 1.0, 0.0)*d;
        case 6:
            return (float3)(0.0, 1.0, 1.0)*d;
        case 7:
            return (float3)(1.0, 0.5, 1.0)*d;
        default:
            return count == 8 ? 1 : .7;
    }
}


kernel void draw_parameter_map(
    const real x_start, const real y_start,
    const real a_min, const real a_max, const real b_min, const real b_max,
    const int skip, const int samples_count,
    
    global real2* samples,
    
    write_only image2d_t result
) {
    
    const int2 id = (int2)(get_global_id(0), get_global_id(1));
    //samples += (id.x * get_global_size(1) + id.y)*samples_count;
    
    //real2 v = (real2)(x_start, y_start);
    
    const real2 par = (real2)(
        a_min + id.x*(a_max - a_min)/get_global_size(0), 
        b_min + (id.y)*(b_max - b_min) / get_global_size(1)
    );
    
    int uniques = 0;
    
    #define D 5e-3
    uint4 color = (uint4)(0, 0, 0, 255);
    if (fabs(par.y - (1 - sqrt(par.x))) < D) {
        color.xyz = (uint3)(255, 0, 0);
    } else if (fabs(par.y - (1 + sqrt(par.x))) < D) { 
        color.xyz = (uint3)(0, 255, 0);
    } else if (fabs(par.y - 2*par.x) < D) { 
        color.xyz = (uint3)(0, 0, 255);
    }
    write_imageui(result, (int2)(id.x, get_global_size(1) - id.y), color);
}

"""


class PhasePortrait:

    def __init__(self, ctx, queue, w, h):
        self.ctx, self.queue, self.w, self.h = ctx, queue, w, h
        self.program = cl.Program(ctx, PHASE_PORTRAIT_KERNEL_SOURCE).build()
        self.image_device = allocate_image(ctx, w, h)
        self.image = np.empty((w, h, 4), dtype=np.uint8)

    def clear(self):
        self.program.clear(self.queue, (self.w, self.h), None, self.image_device)

    def draw(self, m, b, x_min, x_max, y_min, y_max):
        self.clear()
        red = 6
        x_min = -3
        x_max = 3
        y_min = -3
        y_max = 3

        self.program.draw_phase_portrait(self.queue, (self.w // red, self.h // red), None,
                                         real(m), real(b),
                                         real(x_min), real(x_max), real(y_min), real(y_max),
                                         np.int32(20000), np.int32(self.w), np.int32(self.h),
                                         self.image_device)

        cl.enqueue_copy(self.queue, self.image, self.image_device, origin=(0,0), region=(self.w, self.h))

        return self.image

class ParameterMap:

    def __init__(self, ctx, queue, w, h):
        self.ctx, self.queue, self.w, self.h = ctx, queue, w, h
        self.program = cl.Program(ctx, PARAMETER_MAP_KERNEL_SOURCE).build()
        self.image_device = allocate_image(ctx, w, h)
        self.image = np.empty((w, h, 4), dtype=np.uint8)

    def draw(self, x_start, y_start, a_min, a_max, b_min, b_max, skip, samples_count):

        # a_min = 0
        # a_max = 5
        # b_min = 0
        # b_max = 5

        samples_device = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, real_type_size()
                                   )#*2*self.w*self.h*samples_count)

        self.program.draw_parameter_map(self.queue, (self.w, self.h), None,
                                        real(x_start), real(y_start),
                                        real(a_min), real(a_max), real(b_min), real(b_max),
                                        np.int32(skip), np.int32(samples_count),
                                        samples_device, self.image_device)

        cl.enqueue_copy(self.queue, self.image, self.image_device, origin=(0,0), region=(self.w, self.h))

        return self.image



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

        self.param_map = ParameterMap(self.ctx, self.queue, self.w,  self.h)
        self.phase_plot = PhasePortrait(self.ctx, self.queue, self.w, self.h)

        super(App, self).__init__(parent)

        self.setWindowTitle('Task 4')

        self.x_min = 0#-1.2
        self.x_max = 3
        self.y_min = 0#-2
        self.y_max = 3

        self.coord_label = QtGui.QLabel()
        self.image_label = ImageWidget()
        self.image_label.setMouseTracking(True)
        self.image2_label = ImageWidget()
        self.image2_label.setMouseTracking(True)

        def draw_tree_lam(x, y, ev):
            self.pA, self.pA_d = translate(x, self.w, self.x_min, self.x_max), x
            self.pB, self.pB_d = translate(self.h - y, self.h, self.y_min, self.y_max), y
            self.draw_phase_portrait(self.pA, self.pB)
            self.image_label.repaint()

        self.pA, self.pB = None, None
        self.pA_d, self.pB_d = -1, -1

        self.image_label.onMouseMove(lambda x, y, ev: self.coord_label.setText("m = %f; b = %f" % (
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

        self.m = 1.0
        self.b = 1.0

        layout.addWidget(self.coord_label)
        self.slider_m = QtGui.QSlider()
        self.slider_m.setOrientation(QtCore.Qt.Horizontal)
        self.slider_m.setMinimum(-100)
        self.slider_m.setMaximum(100)
        self.slider_m.setSingleStep(.01)
        self.slider_m.sliderMoved.connect(lambda val: (self.draw_phase_portrait(val*self.mul, self.b),))
        layout.addWidget(self.slider_m)

        self.slider_b = QtGui.QSlider()
        self.slider_b.setOrientation(QtCore.Qt.Horizontal)
        self.slider_b.setMinimum(-100)
        self.slider_b.setMaximum(100)
        self.slider_b.setSingleStep(.01)
        self.slider_b.sliderMoved.connect(lambda val: (self.draw_phase_portrait(self.m, val*self.mul),))
        layout.addWidget(self.slider_b)

        self.mblabel = QtGui.QLabel()
        layout.addWidget(self.mblabel)

        # self.slider_m.setDisabled(True)
        self.setLayout(layout)
        self.start = 1.0
        self.mul = .05
        self.image = allocate_image(self.ctx, self.w, self.h )
        self.image_bif = allocate_image(self.ctx, self.w, self.h)

        self.use_param_A = False
        self.skip = 8192
        self.samples = self.w
        self.samples_map = 100

        self.draw_param_map(1, 1)
        self.last_params = (None, None)


    def create_map_image_widget(self):
        pass

    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.key() == QtCore.Qt.Key_Q:
            print("switching param is disabled")
            # self.use_param_A = not self.use_param_A

    def mouseDoubleClickEvent(self, event):
        pass

    def draw_phase_portrait(self, m, b):
        self.m, self.b = m, b
        self.mblabel.setText("m = %f, b = %f" % (self.m, self.b))
        self.image2_label.setPixmap(pixmap_from_raw_image(self.phase_plot.draw(m, b, self.x_min, self.x_max, self.y_min, self.y_max)))

    def draw_param_map(self, x, y):
        self.image_label.setPixmap(pixmap_from_raw_image(self.param_map.draw(x, y, self.x_min, self.x_max, self.y_min, self.y_max,
                                                                             0, 16)))

    def active_param_value(self):
        return self.last_params[0 if not self.use_param_A else 1]

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())


