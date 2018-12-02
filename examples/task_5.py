from dynsys import *

parameter_map_bounds = Bounds(
    -.5, .5,
    0, 2
)

attractor_bounds = Bounds(
    -2, 2,
    -2, 2
)

iter_count = 2 ** 15
draw_count = 1  # 2**15

x0, y0 = -.5, -.5

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

        self.parameter_map = self.makeParameterMap(parameter_map_bounds, map_function_source, var_count=2)
        self.parameter_map_image = ParameterizedImageWidget(parameter_map_bounds, names=("b", "lam"),
                                                            targetColor=QtCore.Qt.white)

        self.attractor = self.makePhasePortrait(attractor_bounds, map_function_source)
        self.attractor_image = ParameterizedImageWidget(attractor_bounds)

        self.parameter_map_image.selectionChanged.connect(self.draw_attractor)

        self.setLayout(
            hStack(
                self.parameter_map_image, self.attractor_image
            )
        )

        self.draw_parameter_map()
        self.draw_attractor(parameter_map_bounds.x_min, parameter_map_bounds.y_min)

    def draw_attractor(self, a, b):
        self.attractor_image.setImage(self.attractor(
            iter_count, a, b, draw_last_points=draw_count
        ))

    def draw_parameter_map(self):
        self.parameter_map_image.setImage(self.parameter_map(
            16, 512, x0, y0
        ))


if __name__ == '__main__':
    Task5().run()
