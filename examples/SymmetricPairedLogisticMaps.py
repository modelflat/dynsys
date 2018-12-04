from dynsys import *
from math import isnan

paramMapBounds = Bounds(
    -.5, 2, -.5, 2
)

attractorBounds = Bounds(
    -2, 2, -2, 2
)

iterations = 2 ** 10
skip = 2 ** 10

systemSource = r"""
real2 userFn(real2, real, real);
real2 userFn(real2 v, real lam1, real lam2) {
    #define EPSILON 0.2
    real xp = lam1 - v.x*v.x + EPSILON*(v.y - v.x); 
    real yp = lam2 - v.y*v.y + EPSILON*(v.x - v.y);
    return (real2)(xp, yp);
}

#define DYNAMIC_COLOR
#define DIVERGENCE_THRESHOLD 1e2
#define DIVERGENCE_COLOR (float4)(.5)
#define DETECTION_PRECISION 1e-2
#define DETECTION_PRECISION_EXPONENT 2
"""


class SymmetricPairedLogisticMaps(SimpleApp):

    def __init__(self):
        super().__init__("Example: Symmetric paired logistic maps - basins of attraction")

        self.paramMap, self.paramMapUi = self.makeParameterMap(
            source=systemSource, variableCount=2,
            spaceShape=paramMapBounds,
            withUi=True,
            uiNames=("λ_1", "λ_2"),
            uiTargetColor=Qt.white
        )

        self.phasePlot, self.phasePlotUi = self.makePhasePlot(
            source=systemSource, paramCount=2,
            spaceShape=attractorBounds,
            withUi=True,
            uiTargetColor=Qt.black
        )

        self.basins, self.basinsUi = self.makeBasinsOfAttraction(
            source=systemSource, paramCount=2,
            spaceShape=attractorBounds,
            withUi=True,
            uiTargetColor=Qt.gray
        )

        self.basinsNumLabel = QLabel()

        def attractorToPhase(val):
            attraction = self.basins.findAttraction(
                targetPoint=val,
                parameters=self.paramMapUi.value(),
                iterations=iterations
            )
            if any(map(isnan, attraction)):
                self.phasePlotUi.setValue(None)
            else:
                self.phasePlotUi.setValue(attraction)

        self.basinsUi.valueChanged.connect(attractorToPhase)
        self.paramMapUi.valueChanged.connect(self.drawBasins)
        self.paramMapUi.valueChanged.connect(self.drawPhasePlot)

        self.setLayout(
            hStack(
                vStack(
                    self.paramMapUi,
                    self.basinsNumLabel
                ),
                hStack(
                    self.phasePlotUi,
                    self.basinsUi
                )
            )
        )
        self.drawParamMap()
        self.drawPhasePlot()
        self.drawBasins()

    def drawBasins(self, params=(1., 1.)):
        img, count = self.basins(
            parameters=params,
            iterations=iterations
        )
        self.basinsUi.setImage(img)
        self.basinsNumLabel.setText("Attractors found: " + str(count))

    def drawPhasePlot(self, params=(1., 1.)):
        self.phasePlotUi.setImage(self.phasePlot(
            parameters=params,
            iterations=iterations,
            skip=skip
        ))

    def drawParamMap(self):
        self.paramMapUi.setImage(self.paramMap(
            variables=(0, 0),
            iterations=80,
            skip=512
        ))


if __name__ == '__main__':
    SymmetricPairedLogisticMaps().run()
