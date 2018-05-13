from dynsys import *


map_function_source = """
real map_function(real x, real a, real b) {
    return a + b*x - x*x*x;
}
"""


class Task3(SimpleApp):

    def __init__(self):
        super().__init__("Task 3")

        self.w, self.h = 512, 512
        self.bounds = Bounds(-1.5, 1.5, -1, 3)
        self.iter_bounds = Bounds(-2, 2, -2, 2)

        self.map = ParameterMap(self.ctx, self.queue, self.w, self.h, self.bounds, map_function_source)
        self.map_image = ParametrizedImageWidget(self.bounds, names=("a", "b"), crosshair_color=QtCore.Qt.white)

        self.iter = CobwebDiagram(self.ctx, self.queue, self.w, self.h, self.iter_bounds, map_function_source,
                                  param_count=2)
        self.iter_image = ParametrizedImageWidget(self.bounds, shape=(False, False), parent=self)

        self.x0 = ObservableValue.makeAndConnect(0)

        self.map_image.selectionChanged.connect(self.draw_iter)

        self.setLayout(qt_hstack(self.map_image, self.iter_image))

        self.draw_iter(1.0, 1.0)
        self.draw_map()

    def draw_iter(self, a, b):
        self.iter(self.x0.value(), 512, a, b, skip_first=0)
        self.iter_image.set_image(self.iter.image)

    def draw_map(self):
        self.map(16, 512, self.x0.value())
        self.map_image.set_image(self.map.image)


if __name__ == '__main__':
    Task3().run()
