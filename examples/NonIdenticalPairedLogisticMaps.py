from dynsys import *

paramMapBoundsZoomed = Bounds(
    .75, 1.14,
    .75, 1.35,
)

paramMapBounds = Bounds(
    -.25, 2.05,
    -.75, 1.7,
)

phasePlotBounds = Bounds(
    -1, 3,
    -1, 3
)

iterations = 2 ** 12
skip = 0

x0, y0 = .0005, .0005

map_function_source = """
real2 map_function(real2, real, real);

real2 map_function(real2 v, real lam, real A) {
    return (real2) (
        1 - lam*v.x*v.x - (-.25)*v.y*v.y,
        1 - A*v.y*v.y - (.375)*v.x*v.x
    );
}

#define DIVERGENCE_THRESHOLD 5
#define DIVERGENCE_COLOR (float4)(.4)
#define system_fn map_function
#define DYNAMIC_COLOR
//#define GENERATE_COLORS
"""


class NonIdenticalPairedLogisticMaps(SimpleApp):

    def __init__(self):
        super().__init__("Example: Non-identical paired logistic maps")

        self.paramMap, self.paramMapUi = self.makeParameterMap(
            source=map_function_source, variableCount=2,
            spaceShape=paramMapBounds,
            withUi=True,
            uiNames=("lam", "A"),
            uiTargetColor=Qt.white
        )

        self.paramMapZoomed, self.paramMapZoomedUi = self.makeParameterMap(
            source=map_function_source, variableCount=2,
            spaceShape=paramMapBoundsZoomed,
            withUi=True,
            uiNames=("lam", "A"),
            uiTargetColor=Qt.white
        )

        self.attractor, self.attractorUi = self.makePhasePlot(
            source=map_function_source, paramCount=2,
            withUi=True
        )

        self.paramMapUi.selectionChanged.connect(self.drawAttractor)
        self.paramMapZoomedUi.selectionChanged.connect(self.drawAttractor)

        self.setLayout(
            hStack(self.paramMapUi, self.paramMapZoomedUi, self.attractorUi)
        )

        self.drawParameterMap()
        self.drawParameterMapZoomed()
        self.drawAttractor(1, 1)

    def drawParameterMap(self):
        self.paramMapUi.setImage(self.paramMap(
            variables=(x0, y0),
            iterations=80,
            skip=512
        ))

    def drawParameterMapZoomed(self):
        self.paramMapZoomedUi.setImage(self.paramMapZoomed(
            variables=(x0, y0),
            iterations=80,
            skip=512
        ))

    def drawAttractor(self, A, lam):
        self.attractorUi.setImage(self.attractor(
            parameters=(A, lam),
            iterations=iterations,
            skip=skip
        ))


if __name__ == '__main__':
    NonIdenticalPairedLogisticMaps().run()
