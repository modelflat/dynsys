from dynsys import *

skipCount = 16

treeSamplesCount = 512
treeSkipCount = 256
treeMaxValue = 10

cobwebBounds = Bounds(
    -2, 2, -2, 2
)

lambdaRange = (
    1, 3
)

mapFunction = """
real userFn(real x, real lam);
real userFn(real x, real lam) {
    return (lam - x*x)*x;
}
"""


class CubicMap(SimpleApp):
    def __init__(self):
        super().__init__("Example: Cubic Map - Bifurcation tree and cobweb diagram")

        self.cobweb, self.cobwebUi = self.makeCobwebDiagram(
            source=mapFunction, paramCount=1,
            spaceShape=cobwebBounds,
            withUi=True,
            uiNames=("x_n", "x_n+1")
        )

        self.bifTree, self.bifTreeUi = self.makeBifurcationTree(
            source=mapFunction, paramCount=1,
            paramRange=lambdaRange,
            withUi=True,
            uiNames=("lambda", None),
        )

        self.x0 = observable(0.1, connectTo=(self.drawTree, self.drawDiagram))
        self.pLambda = observable(2.5, connectTo=self.drawDiagram)
        self.iterations = observable(500, connectTo=self.drawDiagram)

        self.bifTreeUi.valueChanged.connect(self.pLambda.setValue)

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

    def drawDiagram(self, *_):
        self.cobwebUi.setImage(self.cobweb(
            startPoint=self.x0.value(),
            parameters=(self.pLambda.value(),),
            iterations=self.iterations.value(),
            skip=skipCount
        ))

    def drawTree(self, *_):
        self.bifTreeUi.setImage(self.bifTree(
            startPoint=self.x0.value(),
            paramIndex=0,
            paramRange=lambdaRange,
            otherParams=(),
            iterations=treeSamplesCount,
            skip=treeSkipCount,
            maxAllowedValue=treeMaxValue
        ))


if __name__ == '__main__':
    CubicMap().run()
