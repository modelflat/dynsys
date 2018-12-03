from dynsys import *

iter_count = 2 ** 14
skip = 0

paramSurfaceBounds = Bounds(
    -2, 2,
    -2, 2,
)

paramSurfaceSource = """
#define D 1e-4
float3 color_for_point(real2 p) {
    if (fabs( p.x ) < D || fabs(p.y) < D) {
        return (float3)(0, .5, 0);
    }
    return 1.0;
}
"""

phaseBounds = Bounds(
    -4, 4,
    -4, 4,
)

systemFnSource = """
#define STEP (real)(1e-4)
real2 system_fn(real2 v, real lam, real k) {
    real2 p = (real2)(
        (lam + k*v.x*v.x - v.x*v.x*v.x*v.x)*v.y - v.x,
        v.x
    );
    return v + STEP*p;
}
#define DYNAMIC_COLOR
"""


class Task1(SimpleApp):

    def __init__(self):
        super().__init__("Example: ParamSurface + Phase")

        self.paramSurface, self.paramSurfaceUi = self.makeParameterSurface(
            source=paramSurfaceSource,
            spaceShape=paramSurfaceBounds,
            withUi=True,
            uiNames=("lambda", "k"),
            uiTargetColor=Qt.black
        )

        self.attractor, self.attractorUi = self.makePhasePlot(
            source=systemFnSource, paramCount=2,
            spaceShape=phaseBounds,
            withUi=True,
            uiShape=(True, False)
        )

        self.paramSurfaceUi.selectionChanged.connect(
            lambda val, _: self.draw_phase_plot(*val))

        self.setLayout(
            hStack(
                self.paramSurfaceUi, self.attractorUi,
            )
        )

        self.draw_parameter_surface()
        self.draw_phase_plot(1., 1.)

    def draw_parameter_surface(self):
        self.paramSurfaceUi.setImage(self.paramSurface())

    def draw_phase_plot(self, lam, k):
        self.attractorUi.setImage(self.attractor(
            parameters=(lam, k),
            iterations=iter_count,
            skip=skip
        ))


if __name__ == '__main__':
    Task1().run()
