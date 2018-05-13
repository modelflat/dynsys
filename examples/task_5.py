from dynsys import *

map_function_source = """

real2 map_function(real2 v, real b, real lam) {
    real xp = 1 - lam*v.x*v.x - b*v.y; 
    real yp = v.x;
    return (real2)(xp, yp);
}

#define system map_function

//#define DYNAMIC_COLOR
//#define GENERATE_COLORS

#define DIVERGENCE_THRESHOLD 1e3

"""


class Task5(SimpleApp):

    def __init__(self):
        super().__init__("Task 5")
        # self.bounds = Bounds(-1, 1, 0, 2)
        self.bounds = Bounds(-.5, .5, 0, 2)
        self.attr_bounds = Bounds(-2, 2, -2, 2)

        self.iter_count = 2**15
        self.draw_count = 1# 2**15

        self.map = self.makeParameterMap(self.bounds, map_function_source, var_count=2)
        self.map_image = ParametrizedImageWidget(self.bounds, names=("b", "lam"), crosshair_color=QtCore.Qt.white)

        self.attr = self.makePhasePortrait(self.attr_bounds, map_function_source)
        self.attr_image = ParametrizedImageWidget(self.attr_bounds)

        self.map_image.selectionChanged.connect(self.draw_attr)

        self.setLayout(
            qt_hstack(
                self.map_image, self.attr_image
            )
        )

        self.draw_map()

    def draw_attr(self, a, b):
        self.attr_image.set_image(self.attr(
            self.iter_count, a, b, draw_last_points=self.draw_count
        ))

    def draw_map(self):
        x0, y0 = -.5, -.5
        self.map_image.set_image(self.map(
            16, 512, x0, y0
        ))


if __name__ == '__main__':
    Task5().run()
