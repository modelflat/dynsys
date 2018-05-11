import dynsys.iterative_diagram as itd
from dynsys.common import *
from PyQt4 import QtGui

class App(SimpleApp):
    def __init__(self):
        super(App, self).__init__("Iterative Diagram")
        self.w, self.h = 512, 512
        self.diag = itd.IterativeDiagram(self.ctx, self.queue, self.w, self.h, """
        real carrying_function(real x, real lam) {
            return (lam - x*x)*x;
        }
        #define map_function carrying_function
        """)

        self.bounds = itd.Bounds(-2, 2, -2, 2)

        self.x0, self.lam, self.iter_count = .1, 1.2, 1000

        self.iter_diag_image = ParametrizedImageWidget(self.bounds, ["x", "y"], parent=self)

        self.lambda_slider = RealSlider(0, 3, horizontal=False)
        self.lambda_slider.valueChanged.connect(self.setLambda)

        self.x0_slider = RealSlider(-1, 1)
        self.x0_slider.valueChanged.connect(self.setX0)

        self.iter_slider = IntegerSlider(1, 1000)
        self.iter_slider.valueChanged.connect(self.setIters)

        self.info_label = QtGui.QLabel()

        layout = Qt.QVBoxLayout()
        layout.addLayout(qt_hstack_widgets(self.iter_diag_image, self.lambda_slider))
        layout.addLayout(qt_vstack_widgets(self.x0_slider, self.iter_slider))
        layout.addWidget(self.info_label)
        self.setLayout(layout )

        self.draw_diag()

    def setLambda(self, v):
        self.lam = v
        self.draw_diag()

    def setX0(self, v):
        self.x0 = v
        self.draw_diag()

    def setIters(self, v):
        self.iter_count = v
        self.draw_diag()

    def draw_diag(self):
        self.update_info()
        self.diag(self.x0, self.lam, self.iter_count, self.bounds)
        self.iter_diag_image.set_image(self.diag.image)

    def update_info(self):
        self.info_label.setText("x0 = %f, lambda = %f, iter_count = %d" % (self.x0, self.lam, self.iter_count))


if __name__ == '__main__':
    App().run()
