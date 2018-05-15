from dynsys import *

parameter_map_bounds = Bounds(
    -2.5, 2.5,
    -1.5, 1.75,  # j
)

attractor_bounds = Bounds(
    -5, 1.5,
    -5, 1.5
)

iter_count = 2 ** 15
draw_last = 2 ** 14

x0, y0 = .005, .005

map_function_source = """
real2 map_function(real2 v, real S, real J) {
    return (real2) (
        S*v.x - v.y - (v.x*v.x + v.y*v.y),
        J*v.x - (v.x*v.x + v.y*v.y) / 5
    );
}
#define DIVERGENCE_THRESHOLD 5
#define DIVERGENCE_COLOR (float4)(.4)
#define system map_function
#define DYNAMIC_COLOR
"""


class Task7(SimpleApp):

    def __init__(self):
        super().__init__("Task 7")

        self.parameter_map = self.makeParameterMap(parameter_map_bounds, map_function_source,
                                                   width=1024, height=1024,
                                                   var_count=2)
        self.parameter_map_image = ParametrizedImageWidget(parameter_map_bounds, names=["A", "lam"])

        self.attractor = self.makePhasePortrait(attractor_bounds, map_function_source)
        self.attractor_image = ParametrizedImageWidget(attractor_bounds)

        self.parameter_map_image.selectionChanged.connect(self.draw_attractor)

        self.setLayout(
            qt_hstack(self.parameter_map_image, self.attractor_image)
        )

        self.draw_parameter_map()

    def draw_parameter_map(self):
        self.parameter_map_image.set_image(self.parameter_map(
            32, 512, x0, y0
        ))

    def draw_attractor(self, A, lam):
        self.attractor_image.set_image(self.attractor(
            iter_count, A, lam, draw_last_points=draw_last
        ))


if __name__ == '__main__':
    Task7().run()
