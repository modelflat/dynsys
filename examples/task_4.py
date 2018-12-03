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

        self.paramSurface, self.paramSurfaceUi = self.makeParameterSurface(
            source=parameter_surface_color_function,
            spaceShape=parameter_map_bounds,
            withUi=True,
            uiNames=("b", "m"),
            uiTargetColor=Qt.white
        )

        self.attractor, self.attractorUi = self.makePhasePlot(
            source=system_function_source, paramCount=2,
            spaceShape=attractor_bounds,
            withUi=True
        )

        self.paramSurfaceUi.selectionChanged.connect(
            lambda val, _: self.drawAttractor(*val)
        )

        self.setLayout(
            hStack(
                self.paramSurfaceUi, self.attractorUi
            )
        )

        self.drawParamSurface()
        self.drawAttractor(parameter_map_bounds.x_min, parameter_map_bounds.y_min)

    def drawAttractor(self, a, b):
        self.attractorUi.setImage(self.attractor(
            parameters=(a, b),
            iterations=iterations,
            skip=skip,
            gridSparseness=16
        ))

    def drawParamSurface(self):
        self.paramSurfaceUi.setImage(self.paramSurface())



if __name__ == '__main__':
    Task4().run()
