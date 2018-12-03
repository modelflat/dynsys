from dynsys import *

x0 = 0

paramMapBounds = Bounds(
    -1.5, 1.5, -1, 3
)

cobwebBounds = Bounds(
    -2, 2, -2, 2
)

mapFunction = """

real map_function(real, real, real);

real map_function(real x, real a, real b) {
    return a + b*x - x*x*x;
}
"""


class Task3(SimpleApp):

    def __init__(self):
        super().__init__("Example: ParamMap + Cobweb")

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

        self.paramMapUi.selectionChanged.connect(
            lambda val, _: self.drawCobwebDiagram(*val)
        )

        self.setLayout(
            hStack(self.paramMapUi, self.cobwebUi)
        )

        self.drawParamMap()
        self.drawCobwebDiagram(1.0, 1.0)

    def drawCobwebDiagram(self, a, b):
        self.cobwebUi.setImage(self.cobweb(
            startPoint=x0,
            parameters=(a, b),
            iterations=512,
            skip=0
        ))

    def drawParamMap(self):
        self.paramMapUi.setImage(self.paramMap(
            variables=(x0,),
            iterations=80,
            skip=512
        ))


if __name__ == '__main__':
    Task3().run()
