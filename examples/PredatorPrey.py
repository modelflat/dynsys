from dynsys import *

parameterMapBounds = Bounds(
    0, 3, 0, 3
)

phasePlotBounds = Bounds(
    -3, 3, -3, 3
)

iterations = 2 ** 15
skip = 0

systemSource = r"""
real2 userFn(real2, real, real);
real2 userFn(real2 v, real m, real b) {
    #define STEP (real)(4e-4)
    real2 p = (real2)(
        (1 + b*v.x - v.x*v.x)*v.x - v.x*v.y,
        v.y*(v.x - m)
    );
    return v + STEP*p;
}
#define DYNAMIC_COLOR
"""

parameterSurfaceSource = """
float3 userFn(real2);
float3 userFn(real2 p) {
    #define D 5e-3
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

        self.paramSurfaceUi.valueChanged.connect(self.drawAttractor)

        self.setLayout(
            hStack(
                self.paramSurfaceUi, self.attractorUi
            )
        )

        self.drawParamSurface()
        self.drawAttractor((parameterMapBounds.xMin, parameterMapBounds.yMin))

    def drawAttractor(self, params):
        self.attractorUi.setImage(self.attractor(
            parameters=params,
            iterations=iterations,
            skip=skip,
            gridSparseness=16
        ))

    def drawParamSurface(self):
        self.paramSurfaceUi.setImage(self.paramSurface())


if __name__ == '__main__':
    PredatorPrey().run()
