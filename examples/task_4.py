from dynsys import *

parameter_map_bounds = Bounds(
    0, 3,
    0, 3
)

attractor_bounds = Bounds(
    -3, 3,
    -3, 3
)

iterations = 2 ** 15
skip = 0

system_function_source = """
#define STEP (real)(4e-4)
real2 system_fn(real2 v, real m, real b) {
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
        self.parameter_surface_image = ParameterizedImageWidget(parameter_map_bounds.asTuple(), names=("b", "m"),
                                                                targetColor=Qt.white)

        self.attractor = self.makePhasePortrait((512, 512),attractor_bounds.asTuple(), system_function_source, 2)
        self.attractor_image = ParameterizedImageWidget(attractor_bounds.asTuple(), shape=(False, False))

        self.parameter_surface_image.selectionChanged.connect(
            lambda val, _: self.draw_attractor(*val))

        self.setLayout(
            hStack(
                self.parameter_surface_image, self.attractor_image
            )
        )

        self.draw_pararameter_surface()
        self.draw_attractor(parameter_map_bounds.x_min, parameter_map_bounds.y_min)

    def draw_attractor(self, a, b):
        self.attractor_image.setImage(self.attractor(
            a, b, iterations=iterations, skip=skip
        ))

    def draw_pararameter_surface(self):
        self.parameter_surface_image.setImage(self.parameter_surface())



if __name__ == '__main__':
    Task4().run()
