from dynsys import *

paramMapBounds = Bounds(
    -1.5, 1.5, -1, 3
)

cobwebBounds = Bounds(
    -2, 2, -2, 2
)

mapFunction = """
real userFn(real, real, real);
real userFn(real x, real a, real b) {
    return a + b*x - x*x*x;
}
"""


class CubicMap2Param(SimpleApp):

    def __init__(self):
        super().__init__("Example: Cubic Map with 2 parameters")

        self.paramMap, self.paramMapUi = self.makeParameterMap(
            source=mapFunction,
            variableCount=1,
            spaceShape=paramMapBounds,
            withUi=True,
            uiNames=("a", "b"),
            uiTargetColor=Qt.white
        )

        self.cobweb, self.cobwebUi = self.makeCobwebDiagram(
            source=mapFunction,
            paramCount=2,
            spaceShape=cobwebBounds,
            withUi=True,
            uiShape=(False, False)
        )

        self.paramMapUi.valueChanged.connect(self.drawCobwebDiagram)

        self.setLayout(
            hStack(self.paramMapUi, self.cobwebUi)
        )

        self.drawParamMap()
        self.drawCobwebDiagram((1.0, 1.0))

    def drawCobwebDiagram(self, ab):
        self.cobwebUi.setImage(self.cobweb(
            startPoint=0,
            parameters=ab,
            iterations=512,
            skip=0
        ))

    def drawParamMap(self):
        self.paramMapUi.setImage(self.paramMap(
            variables=(0,),
            iterations=80,
            skip=512
        ))


if __name__ == '__main__':
    CubicMap2Param().run()
