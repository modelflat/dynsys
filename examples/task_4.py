from dynsys import *

parameter_map_bounds = Bounds(
    0, 3,
    0, 3
)

attractor_bounds = Bounds(
    -3, 3,
    -3, 3
)

iter_count = 2 ** 15
draw_count = iter_count  # 16

system_function_source = """
#define STEP (real)(4e-4)
real2 system(real2 v, real m, real b) {
    real2 p = (real2)(
        (1 + b*v.x - v.x*v.x)*v.x - v.x*v.y,
        v.y*(v.x - m)
    );
    return v + STEP*p;
}
#define DYNAMIC_COLOR
"""

parameter_surface_color_function = """
#define D 5e-3
float3 color_for_point(real2 p) {
    if (fabs(p.y - (1 - sqrt(p.x))) < D) {
        return (float3)(0, 0, 1);
    }
    if (fabs(p.y - (1 + sqrt(p.x))) < D) { 
        return (float3)(0, 1, 0);
    } 
    if (fabs(p.y - 2*p.x) < D) { 
        return (float3)(1, 0, 0);
    }
    return 0;
}
"""


class Task4(SimpleApp):

    def __init__(self):
        super().__init__("Task 4")

        self.parameter_surface = self.makeParameterSurface(parameter_map_bounds, parameter_surface_color_function)
        self.parameter_surface_image = ParametrizedImageWidget(parameter_map_bounds, names=("b", "m"),
                                                               crosshair_color=QtCore.Qt.white)

        self.attractor = self.makePhasePortrait(attractor_bounds, system_function_source)
        self.attractor_image = ParametrizedImageWidget(attractor_bounds, shape=(False, False))

        self.parameter_surface_image.selectionChanged.connect(self.draw_attractor)

        self.setLayout(
            qt_hstack(
                self.parameter_surface_image, self.attractor_image
            )
        )

        self.draw_pararameter_surface()
        self.draw_attractor(parameter_map_bounds.x_min, parameter_map_bounds.y_min)

    def draw_attractor(self, a, b):
        self.attractor_image.set_image(self.attractor(
            iter_count, a, b, draw_last_points=draw_count
        ))

    def draw_pararameter_surface(self):
        self.parameter_surface_image.set_image(self.parameter_surface())



if __name__ == '__main__':
    Task4().run()
