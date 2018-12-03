from dynsys import *

parameterMapBounds = Bounds(
    0, 3,
    0, 3
)

phasePlotBounds = Bounds(
    -3, 3,
    -3, 3
)

iterations = 2 ** 15
skip = 0

systemSource = r"""
real2 system_fn(real2, real, real);

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

parameterSurfaceSource = """
float3 color_for_point(real2);

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


class PredatorPrey(SimpleApp):

    def __init__(self):
        super().__init__("Example: Predator-Prey Model, phase plot and parameter surface with interesting areas")

        self.paramSurface, self.paramSurfaceUi = self.makeParameterSurface(
            source=parameterSurfaceSource,
            spaceShape=parameterMapBounds,
            withUi=True,
            uiNames=("b", "m"),
            uiTargetColor=Qt.white
        )

        self.attractor, self.attractorUi = self.makePhasePlot(
            source=systemSource, paramCount=2,
            spaceShape=phasePlotBounds,
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
        self.drawAttractor(parameterMapBounds.xMin, parameterMapBounds.yMin)

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
    PredatorPrey().run()
