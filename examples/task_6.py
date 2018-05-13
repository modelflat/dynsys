from dynsys import *

from math import isnan

from dynsys.basins_of_attraction import BasinsOfAttraction

map_function_source = """

#define EPSILON 0.2

real2 map_function(real2 v, real lam1, real lam2) {
    real xp = lam1 - v.x*v.x + EPSILON*(v.y - v.x); 
    real yp = lam2 - v.y*v.y + EPSILON*(v.x - v.y);
    return (real2)(xp, yp);
}

#define system map_function

//#define DYNAMIC_COLOR
#define GENERATE_COLORS

#define DIVERGENCE_THRESHOLD 1e2

"""


class Task6(SimpleApp):

    def __init__(self):
        super().__init__("Task 6")

        self.bounds = Bounds(-.5, 2, -.5, 2)
        self.attr_bounds = Bounds(-2, 2, -2, 2)

        self.iter_count = 2 ** 10
        self.draw_count = 1#2**15

        self.map = self.makeParameterMap(self.bounds, map_function_source, var_count=2)
        self.map_image = ParametrizedImageWidget(self.bounds, names=("lam1", "lam2"), crosshair_color=QtCore.Qt.white)

        sub_w, sub_h = 384, 384

        self.phas = self.makePhasePortrait(self.attr_bounds,map_function_source, width=sub_w, height=sub_h)
        self.phas_image = ParametrizedImageWidget(self.attr_bounds)

        self.attr = BasinsOfAttraction(self.ctx, self.queue, sub_w, sub_h, self.attr_bounds, map_function_source)
        self.attr_image = ParametrizedImageWidget(self.attr_bounds, crosshair_color=QtCore.Qt.gray)

        self.basins_label = Qt.QLabel()

        def attr_to_phase(x, y):
            a, b =  self.map_image.get_selection()
            x_attr, y_attr = self.attr.find_attraction(x, y, self.iter_count, a, b)
            if isnan(x_attr) or isnan(y_attr):
                self.phas_image.set_crosshair_pos(-1, -1)
            else:
                self.phas_image.set_crosshair_pos(
                    *self.attr_bounds.to_integer(x_attr, y_attr,
                                                 sub_w, sub_h, invert_y=False))

        self.attr_image.selectionChanged.connect(attr_to_phase)

        self.map_image.selectionChanged.connect(lambda *args: (self.draw_attr(*args), self.draw_phase(*args)))

        self.setLayout(
            qt_hstack(
                qt_vstack(
                    self.phas_image,
                    self.attr_image
                ),
                qt_vstack(
                    self.map_image,
                    self.basins_label
                )
            )
        )

        self.draw_map()

    def draw_attr(self, a, b):
        img, count = self.attr(self.iter_count, a, b)
        self.attr_image.set_image(img)
        self.basins_label.setText("Attractors found: " + str(count))

    def draw_phase(self, a, b):
        self.phas_image.set_image(self.phas(
            self.iter_count, a, b, draw_last_points=self.draw_count
        ))

    def draw_map(self):
        x0, y0 = 0, 0
        self.map_image.set_image(self.map(
            16, 512, x0, y0
        ))


if __name__ == '__main__':
    Task6().run()