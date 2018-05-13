from dynsys import *

system_function_source = """

#define STEP (real)(1e-4)

real2 system(real2 v, real lam, real k) {
    real2 p = (real2)(
        (lam + k*v.x*v.x - v.x*v.x*v.x*v.x)*v.y - v.x,
        v.x
    );
    return v + STEP*p;
}

#define DYNAMIC_COLOR

"""

parameter_surface_source = """

#define D 1e-4

float3 color_for_point(real2 p) {
    if (fabs( p.x ) < D || fabs(p.y) < D) {
        return (float3)(0, .5, 0);
    }
    return 1.0;
}

"""


class Task1(SimpleApp):

    def __init__(self):
        super().__init__("Task 1")

        self.bounds = Bounds(-3, 3, -3, 3)
        self.attr_bounds = Bounds(-4, 4, -4, 4)

        self.iter_count = 2**15
        self.draw_last = self.iter_count

        self.param_surface = self.makeParameterSurface(self.bounds, parameter_surface_source )
        self.param_surface_image = ParametrizedImageWidget(self.bounds, names=("lam", "k"), crosshair_color=QtCore.Qt.black)

        self.attr = self.makePhasePortrait(self.attr_bounds, system_function_source)
        self.attr_image = ParametrizedImageWidget(self.attr_bounds, shape=(False, False))

        self.param_surface_image.selectionChanged.connect(self.draw_attr)

        self.setLayout(
            qt_hstack(
                self.param_surface_image, self.attr_image,
            )
        )

        self.draw_param_surface()
        self.draw_attr(1., 1.)

    def draw_param_surface(self):
        self.param_surface_image.set_image(self.param_surface())

    def draw_attr(self, lam, k):
        self.attr_image.set_image(self.attr(
            self.iter_count, lam, k, draw_last_points=self.draw_last
        ))


if __name__ == '__main__':
    Task1().run()
