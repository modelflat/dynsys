import dynsys.iterative_diagram as itd
from dynsys.common import *
from PyQt4 import QtGui

class App(SimpleApp):
    def __init__(self):
        super(App, self).__init__("Iterative Diagram")
        self.w, self.h = 512, 512
        self.bounds = itd.Bounds(-1, 1, -1, 1)

        self.diag = itd.IterativeDiagram(self.ctx, self.queue, self.w, self.h, """
        real carrying_function(real x, real lam) {
            return (lam - x*x)*x;
        }
        """)

        self.iter_diag_image = ParametrizedImageWidget(self.bounds, ["x", "y"], parent=self)

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.iter_diag_image)
        self.setLayout(layout)

        self.diag(1.2, 100, self.bounds)
        self.iter_diag_image.set_image(self.diag.image)


if __name__ == '__main__':
    App().run()
