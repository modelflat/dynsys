from dynsys import *

skipCount = 16

treeSamplesCount = 512
treeSkipCount = 256
treeMaxValue = 10

cobwebBounds = Bounds(
    -2, 2,
    -2, 2
)

lambda_bounds = Bounds.x(
    1, 3
)

mapFunction = """
real map_function(real x, real lam);

real map_function(real x, real lam) {
    return (lam - x*x)*x;
}
"""


class BifTreeAndCobweb(SimpleApp):
    def __init__(self):
        super().__init__("Example: BifTree + Cobweb")

        self.cobweb, self.cobwebUi = self.makeCobwebDiagram(
            source=mapFunction, paramCount=1,
            spaceShape=cobwebBounds,
            withUi=True,
            uiNames=("x_n", "x_n+1")
        )

        self.bifTree, self.bifTreeUi = self.makeBifurcationTree(
            source=mapFunction, paramCount=1,
            paramRange=lambda_bounds,
            withUi=True,
            uiNames=("lambda", None),
        )

        self.pLambda = ObservableValue.makeAndConnect(
            2.5, connect_to=self.drawDiagram
        )
        self.iterations = ObservableValue.makeAndConnect(
            500, connect_to=self.drawDiagram
        )
        self.x0 = ObservableValue.makeAndConnect(
            0.1, connect_to=lambda *args: (self.drawTree(), self.drawDiagram())
        )

        self.bifTreeUi.selectionChanged.connect(lambda x, y: self.pLambda.setValue(x))

        self.iterationsSlider, iterationsSliderUi = createSlider(
            "integer", (1, 1000),
            withValue=self.iterations.value(),
            connectTo=self.iterations.setValue
        )

        self.x0Slider, x0SliderUi = createSlider(
            "real", (-1.2, 1.2),
            withValue=self.x0.value(),
            connectTo=self.x0.setValue
        )

        self.setLayout(
            vStack(
                hStack(self.bifTreeUi, self.cobwebUi),
                vStack(
                    x0SliderUi,
                    iterationsSliderUi,
                )
            )
        )

        self.drawDiagram()
        self.drawTree()

    def drawDiagram(self, *args):
        self.cobwebUi.setImage(self.cobweb(
            startPoint=self.x0.value(),
            parameters=(self.pLambda.value(),),
            iterations=self.iterations.value(),
            skip=skipCount
        ))

    def drawTree(self, *args):
        self.bifTreeUi.setImage(self.bifTree(
            startPoint=self.x0.value(),
            paramIndex=0,
            paramRange=lambda_bounds.asTuple(),
            otherParams=(),
            iterations=treeSamplesCount,
            skip=treeSkipCount,
            maxAllowedValue=treeMaxValue
        ))


if __name__ == '__main__':
    BifTreeAndCobweb().run()
