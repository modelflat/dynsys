from dynsys import *

x0 = 0

parameter_map_bounds = Bounds(
    -1.5, 1.5, -1, 3
)

cobweb_diagram_bounds = Bounds(
    -2, 2, -2, 2
)

map_function_source = """
real map_function(real x, real a, real b) {
    return a + b*x - x*x*x;
}
"""


class Task3(SimpleApp):

    def __init__(self):
        super().__init__("Task 3")

        self.parameter_map = self.makeParameterMap(parameter_map_bounds, map_function_source)
        self.parameter_map_image = ParametrizedImageWidget(parameter_map_bounds, names=("a", "b"),
                                                           crosshair_color=QtCore.Qt.white)

        self.cobweb_diagram = self.makeCobwebDiagram(cobweb_diagram_bounds, map_function_source, param_count=2)
        self.cobweb_diagram_image = ParametrizedImageWidget(cobweb_diagram_bounds, shape=(False, False))

        self.parameter_map_image.selectionChanged.connect(self.draw_cobweb_diagram)

        self.setLayout(
            qt_hstack(self.parameter_map_image, self.cobweb_diagram_image)
        )

        self.draw_parameter_map()
        self.draw_cobweb_diagram(1.0, 1.0)

    def draw_cobweb_diagram(self, a, b):
        self.cobweb_diagram_image.set_image(self.cobweb_diagram(
            x0, 512, a, b, skip_first=0
        ))

    def draw_parameter_map(self):
        self.parameter_map_image.set_image(self.parameter_map(
            16, 512, x0
        ))


if __name__ == '__main__':
    Task3().run()
