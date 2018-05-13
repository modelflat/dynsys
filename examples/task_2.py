from dynsys import *

map_function = """
real map_function(real x, real lam) {
    return (lam - x*x)*x;
}
"""


class Task2(SimpleApp):
    def __init__(self):
        super().__init__("Task 2")

        self.w, self.h = 512, 512
        self.skip_count = 16
        self.bounds = Bounds(-2, 2, -2, 2)
        self.lambda_bounds = Bounds.x(1.9, 3)

        self.info_label = QtGui.QLabel()

        self.diag = CobwebDiagram(self.ctx, self.queue, self.w, self.h, self.bounds, map_function)
        self.tree = BifurcationTree(self.ctx, self.queue, self.w, self.h, map_function)

        self.diag_image = ParametrizedImageWidget(self.bounds, shape=(False, False), parent=self)
        self.tree_image = ParametrizedImageWidget(self.lambda_bounds, names=("lambda", ""), shape=(True, False),
                                                  parent=self)

        self.lam = ObservableValue.makeAndConnect(2.46, connect_to=self.draw_diag)
        self.iter_count = ObservableValue.makeAndConnect(500, connect_to=self.draw_diag)
        self.x0 = ObservableValue.makeAndConnect(0.1, connect_to=lambda *args: (self.draw_tree(), self.draw_diag()))

        self.tree_image.selectionChanged.connect(lambda x, y: self.lam.setValue(x))
        self.iter_slider = IntegerSlider.makeAndConnect(1, 1000, self.iter_count.value(), connect_to=self.iter_count.setValue)
        self.x0_slider = RealSlider.makeAndConnect(-1.2, 1.2, self.x0.value(), connect_to=self.x0.setValue)

        self.setLayout(
            qt_vstack(
                qt_hstack(self.tree_image, self.diag_image),
                qt_vstack(
                    self.x0_slider,
                    self.iter_slider,
                    self.info_label
                )
            )
        )

    def draw_diag(self, *args):
        self.update_info()
        self.diag(self.x0.value(), self.iter_count.value(), self.lam.value(), skip_first=self.skip_count, bounds=self.bounds)
        self.diag_image.set_image(self.diag.image)

    def draw_tree(self, *args):
        self.update_info()
        self.tree(self.x0.value(), 512, self.lambda_bounds.x_min, self.lambda_bounds.x_max, skip=256,
                  max_allowed_value=10)
        self.tree_image.set_image(self.tree.image)

    def update_info(self):
        self.info_label.setText("x0 = %f, lambda = %f, iter_count = %d" % (self.x0.value(), self.lam.value(), self.iter_count.value()))


if __name__ == '__main__':
    Task2().run()
