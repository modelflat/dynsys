from dynsys import *

paramMapBounds = Bounds(
    -.5, .5,
    0, 2
)

phasePlotBounds = Bounds(
    -2, 2,
    -2, 2
)

iterations = 2 ** 15
skip = 2 ** 14

systemSource = """
real2 userFn(real2, real, real);
real2 userFn(real2 v, real b, real lam) {
    real xp = 1 - lam*v.x*v.x - b*v.y; 
    real yp = v.x;
    return (real2)(xp, yp);
}

#define DIVERGENCE_THRESHOLD 1e3
"""


class HenonMap(SimpleApp):

    def __init__(self):
        super().__init__("Example: Henon Map - parameter map and attractor")

        self.paramMap, self.paramMapUi = self.makeParameterMap(
            source=systemSource, variableCount=2,
            spaceShape=paramMapBounds,
            withUi=True,
            uiNames=("b", "lam"),
            uiTargetColor=Qt.white
        )

        self.attractor, self.attractorUi = self.makePhasePlot(
            source=systemSource, paramCount=2,
            spaceShape=phasePlotBounds,
            withUi=True
        )

        self.paramMapUi.valueChanged.connect(self.drawPhasePlot)

        self.setLayout(
            hStack(
                self.paramMapUi, self.attractorUi
            )
        )

        self.drawParamMap()
        self.drawPhasePlot((paramMapBounds.xMin, paramMapBounds.yMin))

    def drawPhasePlot(self, ab):
        self.attractorUi.setImage(self.attractor(
            parameters=ab,
            iterations=iterations,
            skip=skip
        ))

    def drawParamMap(self):
        self.paramMapUi.setImage(self.paramMap(
            variables=(-.5, -.5),
            iterations=16,
            skip=512
        ))


if __name__ == '__main__':
    HenonMap().run()
