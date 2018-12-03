from dynsys import *

parameter_map_bounds = Bounds(
    -.5, .5,
    0, 2
)

attractor_bounds = Bounds(
    -2, 2,
    -2, 2
)

iterations = 2 ** 15
skip = 2**14

x0, y0 = -.5, -.5

map_function_source = """
real2 map_function(real2, real, real);

real2 map_function(real2 v, real b, real lam) {
    real xp = 1 - lam*v.x*v.x - b*v.y; 
    real yp = v.x;
    return (real2)(xp, yp);
}

#define system_fn map_function
//#define DYNAMIC_COLOR
//#define GENERATE_COLORS
#define DIVERGENCE_THRESHOLD 1e3
"""


class Task5(SimpleApp):

    def __init__(self):
        super().__init__("Task 5")

        self.paramMap, self.paramMapUi = self.makeParameterMap(
            source=map_function_source, variableCount=2,
            spaceShape=parameter_map_bounds,
            withUi=True,
            uiNames=("b", "lam"),
            uiTargetColor=Qt.white
        )

        self.attractor, self.attractorUi = self.makePhasePlot(
            source=map_function_source, paramCount=2,
            spaceShape=attractor_bounds,
            withUi=True
        )

        self.paramMapUi.selectionChanged.connect(self.drawAttractor)

        self.setLayout(
            hStack(
                self.paramMapUi, self.attractorUi
            )
        )

        self.drawParamMap()
        self.drawAttractor(parameter_map_bounds.x_min, parameter_map_bounds.y_min)

    def drawAttractor(self, a, b):
        self.attractorUi.setImage(self.attractor(
            parameters=(a, b),
            iterations=iterations,
            skip=skip
        ))

    def drawParamMap(self):
        self.paramMapUi.setImage(self.paramMap(
            variables=(x0, y0),
            iterations=16,
            skip=512
        ))


if __name__ == '__main__':
    Task5().run()
