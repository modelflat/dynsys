from dynsys import *

parameter_map_bounds_zoomed = Bounds(
    .75, 1.14,
    .75, 1.35,
)

parameter_map_bounds = Bounds(
    -.25, 2.05,
    -.75, 1.7,
)

attractor_bounds = Bounds(
    -.5, 1.5,
    -1, 1.5
)

iter_count = 2 ** 12 
draw_last = 2 ** 10

x0, y0 = .0005, .0005

map_function_source = """
real2 map_function(real2 v, real lam, real A) {
    return (real2) (
        1 - lam*v.x*v.x - (-.25)*v.y*v.y,
        1 - A*v.y*v.y - (.375)*v.x*v.x
    );
}
#define DIVERGENCE_THRESHOLD 5
#define DIVERGENCE_COLOR (float4)(.4)
#define system map_function
#define DYNAMIC_COLOR
//#define GENERATE_COLORS
"""


class Task7(SimpleApp):

    def __init__(self):
        super().__init__("Task 7")

        self.parameter_map = self.makeParameterMap(parameter_map_bounds, map_function_source,
                                                   variableCount=2)
        self.parameter_map_image = ParameterizedImageWidget(parameter_map_bounds, names=["lam", "A"],
                                                            targetColor=QtCore.Qt.white)

        self.parameter_map_zoomed = self.makeParameterMap(parameter_map_bounds_zoomed, map_function_source,
                                                          width=512, height=512,
                                                          variableCount=2)
        self.parameter_map_zoomed_image = ParameterizedImageWidget(parameter_map_bounds_zoomed, names=["lam", "A"],
                                                                   targetColor=QtCore.Qt.white)

        self.attractor = self.makePhasePlot(attractor_bounds, map_function_source)
        self.attractor_image = ParameterizedImageWidget(attractor_bounds)

        self.parameter_map_image.selectionChanged.connect(self.draw_attractor)
        self.parameter_map_zoomed_image.selectionChanged.connect(self.draw_attractor)

        self.setLayout(
            hStack(self.parameter_map_zoomed_image,
                   vStack(
                          self.attractor_image,
                          self.parameter_map_image
                      ))
        )

        self.draw_parameter_map()
        self.draw_parameter_map_zoomed()

    def draw_parameter_map(self):
        self.parameter_map_image.setImage(self.parameter_map(
            80, 512, x0, y0
        ))

    def draw_parameter_map_zoomed(self):
        self.parameter_map_zoomed_image.setImage(self.parameter_map_zoomed(
            80, 512, x0, y0
        ))

    def draw_attractor(self, A, lam):
        self.attractor_image.setImage(self.attractor(
            iter_count, A, lam, draw_last_points=draw_last
        ))


if __name__ == '__main__':
    Task7().run()
