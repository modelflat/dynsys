from dynsys import *
from dynsys.ui.slider_widgets import createSlider

iterations = 2 ** 15
skip = iterations // 10 * 9

parameter_surface_bounds = Bounds(
    0, .5,
    0, .5,
)
rBounds = (0, 3)

parameter_surface_source = """
#define D 1e-4
float3 color_for_point(real2 p) {
    if (fabs(p.x - .25) < .01 && fabs(p.y - 0.15) < .01) {
        return (float3)(0, .5, 0);
    }
    return 1.0;
}
"""

phase_plot_bounds = (
    -6, 7,
    0, 10,
    -6, 7,
)

imageShape = (
    128, 128, 128
)

system_function_source = """
#define STEP (real)(1e-2)

// #define T real2
#define T real3

T system_fn(T v, real a, real b, real r) {
    T p = (T)(
        -v.z - v.y, 
        b + (v.x - r)*v.y,
        v.x + a*v.z
        
    );
    return v + STEP*p;
}
#define DYNAMIC_COLOR
"""


class Task1(SimpleApp):

    def __init__(self):
        super().__init__("Task 1")

        self.abSurface = self.makeParameterSurface(parameter_surface_bounds, parameter_surface_source)
        self.abSurfaceUi = ParameterizedImageWidget(parameter_surface_bounds.asTuple(), names=("a", "b"), targetColor=Qt.black)

        self.attr = self.makePhasePortrait(imageShape, phase_plot_bounds, system_function_source, 3)

        self.attr_image = Image3D(phase_plot_bounds)
        # self.attr_image = Image2D(spaceShape=phase_plot_bounds)

        self.rSlider, rSliderUi = createSlider(
            "real", rBounds,
            withLabel="r = {}", labelPosition="top",
            withValue=2.5,
            connectTo=lambda r: self.draw_phase_plot(*self.abSurfaceUi._imageWidget.targetReal(), r)
        )

        self.abSurfaceUi.selectionChanged.connect(
            lambda params, _: self.draw_phase_plot(*params, self.rSlider.value())
        )

        self.setLayout(
            hStack(
                vStack(rSliderUi, self.abSurfaceUi),
                self.attr_image,
            )
        )
        self.attr_image.setFixedSize(512, 512)

        self.draw_parameter_surface()
        self.abSurfaceUi._imageWidget.setTargetReal((.25, .155))
        self.draw_phase_plot(.25, .155, 2.5)

    def draw_parameter_surface(self):
        self.abSurfaceUi.setImage(self.abSurface())

    def draw_phase_plot(self, a, b, r, skip=skip):
        import time
        t = time.perf_counter()
        self.attr_image.setTexture(self.attr(a, b, r, sparse=8, iterations=iterations, skip=skip))
        print( "%.3f s" % (time.perf_counter() - t,))


if __name__ == '__main__':
    Task1().run()
