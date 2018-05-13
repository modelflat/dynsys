from dynsys import *


map_function_source = """

real2 map_function(real2 v, real A, real lam) {
    return (real2) (
        1 - lam*v.x*v.x - (-0.25)*v.y*v.y,
        1 - A*v.y*v.y - (0.375)*v.x*v.x
    );
}

//#define DIVERGENCE_THRESHOLD 1e100

#define system map_function
#define DYNAMIC_COLOR

"""


class Task7(SimpleApp):

    def __init__(self):
        super().__init__("Task 7")

        self.bounds = Bounds(-.8, 5, -.3, 2.2)
        self.attr_bounds = Bounds(-3, 3, -3, 3)

        self.iter_count = 2**15
        self.draw_last = 2**15

        self.map = self.makeParameterMap(self.bounds, map_function_source, var_count=2)
        self.map_image = ParametrizedImageWidget(self.bounds, names=["A", "lam"])

        self.attr = self.makePhasePortrait(self.attr_bounds, map_function_source)
        self.attr_image = ParametrizedImageWidget(self.bounds)

        self.map_image.selectionChanged.connect(self.draw_attr)

        self.setLayout(
            qt_hstack(self.map_image, self.attr_image)
        )

        self.draw_map()

    def draw_map(self):
        self.map_image.set_image(self.map(
            16, 512, .5, .5
        ))

    def draw_attr(self, A, lam):
        self.attr_image.set_image(self.attr(
            self.iter_count, A, lam, draw_last_points=self.draw_last
        ))


if __name__ == '__main__':
    Task7().run()
