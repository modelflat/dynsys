from dynsys import *

iterations = 2 ** 14
skip = 0

paramSurfaceBounds = Bounds(
    -2, 2, -2, 2,
)

phaseBounds = Bounds(
    -4, 4, -4, 4,
)

paramSurfaceSource = r"""
float3 userFn(real2);
float3 userFn(real2 p) {
    #define D 1e-4
    if (fabs( p.x ) < D || fabs(p.y) < D) {
        return (float3)(0, .5, 0);
    }
    return 1.0;
}
"""

systemFnSource = r"""
real2 userFn(real2, real, real);
real2 userFn(real2 v, real lam, real k) {
    #define STEP (real)(1e-4)
    real2 p = (real2)(
        (lam + k*v.x*v.x - v.x*v.x*v.x*v.x)*v.y - v.x,
        v.x
    );
    return v + STEP*p;
}
#define DYNAMIC_COLOR
"""


class LocalOscillator(SimpleApp):

    def __init__(self):
        super().__init__("Example: Local Oscillator Phase Plot")

        self.paramSurface, self.paramSurfaceUi = self.makeParameterSurface(
            source=paramSurfaceSource,
            spaceShape=paramSurfaceBounds,
            withUi=True,
            uiNames=("λ", "k"),
            uiTargetColor=Qt.black
        )

        self.attractor, self.attractorUi = self.makePhasePlot(
            source=systemFnSource, paramCount=2,
            spaceShape=phaseBounds,
            withUi=True,
        )

        self.paramSurfaceUi.valueChanged.connect(self.drawPhasePlot)

        self.setLayout(
            hStack(self.paramSurfaceUi, self.attractorUi)
        )

        self.drawParameterSurface()
        self.drawPhasePlot()

    def drawParameterSurface(self):
        self.paramSurfaceUi.setImage(self.paramSurface())

    def drawPhasePlot(self, params=(1., 1.)):
        self.attractorUi.setImage(self.attractor(
            parameters=params,
            iterations=iterations,
            skip=skip
        ))


if __name__ == '__main__':
    LocalOscillator().run()
