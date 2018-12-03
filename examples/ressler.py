from dynsys import *
from dynsys.ui.SliderWidgets import createSlider

iterations = 5 * 10**4
skip = iterations // 100 * 95

paramSurfaceBounds = Bounds(
    0, .5,
    0, .5,
)
rBounds = (0, 3)

paramSurfaceFn = """
#define D 1e-4
float3 color_for_point(real2 p) {
    if (fabs(p.x - .25f) < .01 && fabs(p.y - 0.15f) < .01) {
        return (float3)(0, .5, 0);
    }
    return 1.0;
}
"""

phaseBounds = (
    -6, 7,
    0, 10,
    -6, 7,
)

imageShape = (
    128,
    128,
    128
)

systemFn = """
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


class Ressler(SimpleApp):

    def __init__(self):
        super().__init__("Example: 3D Ressler system")

        self.abSurface, self.abSurfaceUi = self.makeParameterSurface(
            source=paramSurfaceFn,
            spaceShape=paramSurfaceBounds,
            withUi=True,
            uiNames=("a", "b"),
            uiTargetColor=Qt.black
        )

        self.attractor, self.attractorUi = self.makePhasePlot(
            source=systemFn, paramCount=3,
            spaceShape=phaseBounds,
            imageShape=imageShape,
            backColor=(0.0, 0.0, 0.0, 0.0),
            withUi=False
        ), Image3D(phaseBounds)

        self.rSlider, rSliderUi = createSlider(
            "real", rBounds,
            withLabel="r = {}", labelPosition="top",
            withValue=2.5,
            connectTo=lambda r: self.draw_phase_plot(*self.abSurfaceUi.value(), r)
        )

        self.abSurfaceUi.selectionChanged.connect(
            lambda params, _: self.draw_phase_plot(*params, self.rSlider.value())
        )

        self.setLayout(
            vStack(rSliderUi,
                   hStack(self.abSurfaceUi, self.attractorUi)
            )
        )
        self.attractorUi.setFixedSize(512, 512)

        self.draw_parameter_surface()
        self.abSurfaceUi.setValue((.25, .155))
        self.draw_phase_plot(.25, .155, 2.5)

    def draw_parameter_surface(self):
        self.abSurfaceUi.setImage(self.abSurface())

    def draw_phase_plot(self, a, b, r, skip=skip):
        import time
        t = time.perf_counter()
        self.attractorUi.setTexture(self.attractor(
            parameters=(a, b, r),
            iterations=iterations,
            skip=skip,
            gridSparseness=8
        ))
        self.setWindowTitle("%s | Last draw time: %d ms" % ("Ressler", int(1000*(time.perf_counter() - t))))


if __name__ == '__main__':
    Ressler().run()
