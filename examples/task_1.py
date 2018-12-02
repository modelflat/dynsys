from dynsys import *

iter_count = 2 ** 14
skip = 0

parameter_surface_bounds = Bounds(
    -2, 2,
    -2, 2,
)

parameter_surface_source = """
#define D 1e-4
float3 color_for_point(real2 p) {
    if (fabs( p.x ) < D || fabs(p.y) < D) {
        return (float3)(0, .5, 0);
    }
    return 1.0;
}
"""

phase_plot_bounds = Bounds(
    -4, 4,
    -4, 4,
)

system_function_source = """
#define STEP (real)(1e-4)
real2 system_fn(real2 v, real lam, real k) {
    real2 p = (real2)(
        (lam + k*v.x*v.x - v.x*v.x*v.x*v.x)*v.y - v.x,
        v.x
    );
    return v + STEP*p;
}
// #define DYNAMIC_COLOR
"""


class Task1(SimpleApp):

    def __init__(self):
        super().__init__("Task 1")

        self.param_surface = self.makeParameterSurface(parameter_surface_bounds, parameter_surface_source)
        self.param_surface_image = ParameterizedImageWidget(parameter_surface_bounds.asTuple(), names=("lam", "k"),
                                                            targetColor=Qt.black)

        self.attr = self.makePhasePortrait((512, 512), phase_plot_bounds, system_function_source, 2)
        self.attr_image = ParameterizedImageWidget(phase_plot_bounds.asTuple(), shape=(True, False))

        self.param_surface_image.selectionChanged.connect(
            lambda val, _: self.draw_phase_plot(*val))

        self.setLayout(
            hStack(
                self.param_surface_image, self.attr_image,
            )
        )

        self.draw_parameter_surface()
        self.draw_phase_plot(1., 1.)

    def draw_parameter_surface(self):
        self.param_surface_image.setImage(self.param_surface())

    def draw_phase_plot(self, lam, k):
        self.attr_image.setImage(self.attr(lam, k, iterations=iter_count, skip=skip))


if __name__ == '__main__':
    Task1().run()
