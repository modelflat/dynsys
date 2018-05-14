from dynsys import *

skip_count = 16

tree_samples_count = 512
tree_skip_count = 256
tree_max_value = 10

cobweb_bounds = Bounds(
    -2, 2,
    -2, 2
)

lambda_bounds = Bounds.x(
    1.9, 3
)

map_function_source = """
real map_function(real x, real lam) {
    return (lam - x*x)*x;
}
"""


class Task2(SimpleApp):
    def __init__(self):
        super().__init__("Task 2")

        self.cobweb_diagram = self.makeCobwebDiagram(cobweb_bounds, map_function_source)
        self.cobweb_diagram_image = ParametrizedImageWidget(cobweb_bounds, shape=(False, False))

        self.bifurcation_tree = self.makeBifurcationTree(map_function_source)
        self.bifurcation_tree_image = ParametrizedImageWidget(lambda_bounds, names=("lambda", ""),
                                                              shape=(True, False))

        self.p_lambda = \
            ObservableValue.makeAndConnect(2.5, connect_to=self.draw_diag)
        self.iter_count = \
            ObservableValue.makeAndConnect(500, connect_to=self.draw_diag)
        self.x0 = \
            ObservableValue.makeAndConnect(0.1, connect_to=lambda *args: (self.draw_tree(), self.draw_diag()))

        self.bifurcation_tree_image.selectionChanged.connect(lambda x, y: self.p_lambda.setValue(x))

        self.iter_count_slider = IntegerSlider.makeAndConnect(1, 1000, self.iter_count.value(),
                                                              connect_to=self.iter_count.setValue)
        self.x0_slider = RealSlider.makeAndConnect(-1.2, 1.2, self.x0.value(), connect_to=self.x0.setValue)

        self.setLayout(
            qt_vstack(
                qt_hstack(self.bifurcation_tree_image, self.cobweb_diagram_image),
                qt_vstack(
                    self.x0_slider,
                    self.iter_count_slider,
                )
            )
        )

    def draw_diag(self, *args):
        self.cobweb_diagram_image.set_image(self.cobweb_diagram(
            self.x0.value(), self.iter_count.value(), self.p_lambda.value(), skip_first=skip_count
        ))

    def draw_tree(self, *args):
        self.bifurcation_tree_image.set_image(self.bifurcation_tree(
            self.x0.value(), tree_samples_count, lambda_bounds.x_min, lambda_bounds.x_max,
            skip=tree_skip_count, max_allowed_value=tree_max_value
        ))


if __name__ == '__main__':
    Task2().run()
