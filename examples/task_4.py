from dynsys import *
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

        self.w, self.h = 512, 512
        self.param_map_bounds = Bounds(0, 3, 0, 3)
        self.attractor_boudns = Bounds(-3, 3, -3, 3)

        self.iter_count = 2**15
        self.draw_last = self.iter_count # 16

        self.param_surface = self.makeParameterSurface(self.param_map_bounds, parameter_surface_color_function)
        self.param_surface_image = ParametrizedImageWidget(self.param_map_bounds, names=("b", "m"),
                                                           crosshair_color=QtCore.Qt.white)

        self.attractor = self.makePhasePortrait(self.attractor_boudns, system_function_source)
        self.attractor_image = ParametrizedImageWidget(self.attractor_boudns, shape=(False, False))

        self.param_surface_image.selectionChanged.connect(self.draw_attr)

        self.setLayout(
            qt_hstack(
                self.param_surface_image, self.attractor_image
            )
        )

        self.draw_pars()
        self.draw_attr(1., 1.)

    def draw_attr(self, a, b):
        self.attractor_image.set_image(self.attractor(
            self.iter_count, a, b, draw_last_points=self.draw_last
        ))

    def draw_pars(self):
        self.param_surface_image.set_image( self.param_surface() )



if __name__ == '__main__':
    Task4().run()
