from dynsys import *

from math import isnan

parameter_map_bounds = Bounds(
    -.5, 2,
    -.5, 2
)

attractor_bounds = Bounds(
    -2, 2,
    -2, 2
)

iterations = 2 ** 10
skip = 2 ** 10

x0, y0 = 0, 0

map_function_source = """
real2 map_function(real2, real, real);

#define EPSILON 0.2
real2 map_function(real2 v, real lam1, real lam2) {
    real xp = lam1 - v.x*v.x + EPSILON*(v.y - v.x); 
    real yp = lam2 - v.y*v.y + EPSILON*(v.x - v.y);
    return (real2)(xp, yp);
}
#define system_fn map_function

#define DYNAMIC_COLOR
//#define GENERATE_COLORS
#define DIVERGENCE_THRESHOLD 1e2
#define DIVERGENCE_COLOR (float4)(.5)
#define DETECTION_PRECISION 1e-2
#define DETECTION_PRECISION_EXPONENT 2

"""


class Task6(SimpleApp):

    def __init__(self):
        super().__init__("Task 6")

        self.paramMap, self.paramMapUi = self.makeParameterMap(
            source=map_function_source, variableCount=2,
            spaceShape=parameter_map_bounds,
            imageShape=(256, 256),
            withUi=True,
            uiNames=("lam1", "lam2"),
            uiTargetColor=Qt.white
        )

        self.phasePlot, self.phasePlotUi = self.makePhasePlot(
            source=map_function_source, paramCount=2,
            spaceShape=attractor_bounds,
            imageShape=(256, 256),
            withUi=True,
            uiTargetColor=Qt.black
        )

        self.basins, self.basinsUi = self.makeBasinsOfAttraction(
            source=map_function_source, paramCount=2,
            spaceShape=attractor_bounds,
            imageShape=(256, 256),
            withUi=True,
            uiTargetColor=Qt.gray
        )

        self.basinsNumLabel = QLabel()

        def attr_to_phase(val, _):
            attraction = self.basins.findAttraction(
                targetPoint=val,
                parameters=self.paramMapUi.value(),
                iterations=iterations
            )
            if any(map(isnan, attraction)):
                self.phasePlotUi.setValue(None)
            else:
                self.phasePlotUi.setValue(attraction)

        self.basinsUi.selectionChanged.connect(attr_to_phase)

        self.paramMapUi.selectionChanged.connect(
            lambda val, _: (self.drawBasins(*val), self.drawPhasePlot(*val))
        )

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
        self.drawPhasePlot(1., 1.)
        self.drawBasins(1., 1.)

    def drawBasins(self, a, b):
        img, count = self.basins(
            parameters=(a, b),
            iterations=iterations
        )
        self.basinsUi.setImage(img)
        self.basinsNumLabel.setText("Attractors found: " + str(count))

    def drawPhasePlot(self, a, b):
        self.phasePlotUi.setImage(self.phasePlot(
            parameters=(a, b),
            iterations=iterations,
            skip=skip
        ))

    def drawParamMap(self):
        self.paramMapUi.setImage(self.paramMap(
            variables=(x0, y0),
            iterations=80,
            skip=512
        ))


if __name__ == '__main__':
    Task6().run()
