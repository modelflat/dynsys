from dynsys import *

from math import isnan

parameter_map_bounds = Bounds(
    -.5, 2,
    -.5, 2
)

attractor_bounds = Bounds(
    -2, 2,
    -2, 2
)

iter_count = 2 ** 10
draw_count = 2 ** 10

x0, y0 = 0, 0

# sizes of sub-images
sub_w, sub_h = 512, 512

map_function_source = """
#define EPSILON 0.2
real2 map_function(real2 v, real lam1, real lam2) {
    real xp = lam1 - v.x*v.x + EPSILON*(v.y - v.x); 
    real yp = lam2 - v.y*v.y + EPSILON*(v.x - v.y);
    return (real2)(xp, yp);
}
#define system map_function
#define DYNAMIC_COLOR
//#define GENERATE_COLORS
#define DIVERGENCE_THRESHOLD 1e2
#define DIVERGENCE_COLOR (float4)(.5)
#define DETECTION_PRECISION 1e-2
#define DETECTION_PRECISION_EXPONENT 2

"""


class Task6(SimpleApp):

    def __init__(self):
        super().__init__("Task 6")

        self.parameter_map = self.makeParameterMap(parameter_map_bounds, map_function_source, var_count=2)
        self.parameter_map_image = ParameterizedImageWidget(parameter_map_bounds, names=("lam1", "lam2"),
                                                            targetColor=QtCore.Qt.white)

        self.phase_plot = self.makePhasePortrait(attractor_bounds, map_function_source, width=sub_w, height=sub_h)
        self.phase_plot_image = ParameterizedImageWidget(attractor_bounds)

        self.basins_of_attraction = self.makeBasinsOfAttraction(attractor_bounds, map_function_source, width=sub_w, height=sub_h)
        self.basins_of_attraction_image = ParameterizedImageWidget(attractor_bounds, targetColor=QtCore.Qt.gray)

        self.basins_label = Qt.QLabel()

        def attr_to_phase(x, y):
            a, b =  self.parameter_map_image.getTarget()
            x_attr, y_attr = self.basins_of_attraction.find_attraction(x, y, iter_count, a, b)
            if isnan(x_attr) or isnan(y_attr):
                self.phase_plot_image.setTarget(-1, -1)
            else:
                self.phase_plot_image.setTarget(
                    *attractor_bounds.to_integer(x_attr, y_attr, sub_w, sub_h, invert_y=True))

        self.basins_of_attraction_image.selectionChanged.connect(attr_to_phase)

        self.parameter_map_image.selectionChanged.connect(lambda *args: (self.draw_basins(*args), self.draw_phase_plot(*args)))

        self.setLayout(
            hStack(
                vStack(
                    self.parameter_map_image,
                    self.basins_label
                ),
                hStack(
                    self.phase_plot_image,
                    self.basins_of_attraction_image
                )
            )
        )
        self.draw_parameter_map()

    def draw_basins(self, a, b):
        img, count = self.basins_of_attraction(iter_count, a, b)
        self.basins_of_attraction_image.setImage(img)
        self.basins_label.setText("Attractors found: " + str(count))

    def draw_phase_plot(self, a, b):
        self.phase_plot_image.setImage(self.phase_plot(
            iter_count, a, b, draw_last_points=draw_count
        ))

    def draw_parameter_map(self):
        self.parameter_map_image.setImage(self.parameter_map(
            80, 512, x0, y0
        ))


if __name__ == '__main__':
    Task6().run()