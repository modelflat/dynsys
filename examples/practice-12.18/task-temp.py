from dynsys import Bounds, SimpleApp, hStack, vStack, Qt, FLOAT, QLabel, \
    ParameterizedImageWidget, Image2D, Image3D, RealSlider, createSlider
from dynsys.PhasePlotV2 import PhasePlot, OutputConfig

from PyQt5.QtWidgets import QGroupBox

from dynsys.ParameterSurface import ParameterSurface

parameterMapBoundsHG = Bounds(
    .06, .15, .75, .95
)

epsBounds = 0, .5


phaseBounds = (
    -1.5, 1.5,
    -1.5, 1.5,
    -0.1, 1.1,
)

skip = 4 * 10**4
iterations = 512

systemSource = r"""

#define Fz(z) (8.592*(z) - 22*(z)*(z) + 14.408*(z)*(z)*(z))

real3 userFn(real3, real, real, real);
real3 userFn(real3 v, real h, real g, real eps) {
    #define STEP (real)(2e-3)
    real3 p = (real3)(
        2.0f*h*v.x + v.y - g*v.z, 
        -v.x,
        (v.x - Fz(v.z)) / eps
    );
    return v + STEP*p;
}
#define DYNAMIC_COLOR
#ifdef DEFAULT_ENTITY_COLOR
//#undef DEFAULT_ENTITY_COLOR
//#define DEFAULT_ENTITY_COLOR (float4)(0.0, 1.0, 0.0, 1.0)
#endif
#define GENERATE_COLORS
"""


parameterSurfaceSource = """
float3 userFn(real2);
float3 userFn(real2 p) {
    #define D 5e-3
    return 1;
}
"""


class Task1(SimpleApp):

    def __init__(self):
        super().__init__("Task 1")

        self.paramSurface1 = ParameterSurface(self.ctx, self.queue,
                                          (256, 256), parameterMapBoundsHG.asTuple(),
                                          parameterSurfaceSource, typeConfig=FLOAT)
        self.paramSurfaceUi1 = ParameterizedImageWidget(bounds=parameterMapBoundsHG.asTuple(),
                                                        names=("h", "g"),
                                                        shape=(True, True),
                                                        textureShape=(256, 256))

        self.epsSlider, self.epsSliderWidget = createSlider("real", epsBounds,
                                      withLabel="Epsilon = {}",
                                      labelPosition="top",
                                      withValue=0.2,
                                      connectTo=self.drawAttractor)
        self.x0Slider, self.x0SliderWidget = createSlider("real", epsBounds,
                                                            withLabel="x0 = {}",
                                                            labelPosition="top",
                                                            withValue=0.2,
                                                            connectTo=self.drawAttractor)
        self.y0Slider, self.y0SliderWidget = createSlider("real", epsBounds,
                                                            withLabel="y0 = {}",
                                                            labelPosition="top",
                                                            withValue=0.2,
                                                            connectTo=self.drawAttractor)
        self.z0Slider, self.z0SliderWidget = createSlider("real", epsBounds,
                                                            withLabel="z0 = {}",
                                                            labelPosition="top",
                                                            withValue=0.2,
                                                            connectTo=self.drawAttractor)

        self.skipSlider, self.skipSliderWidget = createSlider(
            "int", (0, 40000), labelPosition="top", withValue=skip, withLabel="skip = {}",
            connectTo=self.drawAttractor
        )
        self.iterSlider, self.iterSliderWidget = createSlider(
            "int", (0, 40000), labelPosition="top", withValue=iterations, withLabel="iter = {}",
            connectTo=self.drawAttractor
        )

        self.attr = PhasePlot(self.ctx, self.queue, (8, 8, 8), phaseBounds,
                              systemSource, 3, 3,
                              typeConfig=FLOAT,
                              outputConf=OutputConfig(
                                  shapeXY=(256, 256),
                                  shapeXZ=(256, 256),
                                  shapeYZ=(256, 256),
                                  shapeXYZ=(256, 256, 256),
                              ))
        self.attrXY = Image2D(targetShape=(False, False), spaceShape=phaseBounds[:2])
        self.attrYZ = Image2D(targetShape=(False, False), spaceShape=phaseBounds[2:4])
        self.attrXZ = Image2D(targetShape=(False, False), spaceShape=phaseBounds[4:])
        self.attrXYZ = Image3D(spaceShape=phaseBounds)

        self.paramSurfaceUi1.valueChanged.connect(self.drawAttractor)

        self.v0group = QGroupBox()
        self.v0group.setLayout(
            vStack(self.x0Slider, self.y0Slider, self.z0Slider)
        )

        self.setLayout(
            vStack(
                hStack(
                    vStack(QLabel("X-Y"), self.attrXY),
                    vStack(QLabel("Y-Z"), self.attrYZ),
                    vStack(QLabel("X-Z"), self.attrXZ),
                ),
                hStack(
                    vStack(
                        self.epsSliderWidget, self.paramSurfaceUi1,
                        self.iterSlider, self.skipSlider,
                    ),
                    # self.v0group,

                    self.attrXYZ
                )
            )
        )
        self.attrXYZ.setFixedSize(400, 400)

        self.paramSurfaceUi1.setValue((0.09, 0.87))
        self.drawParamSurface()
        self.drawAttractor()

    def drawAttractor(self, *_):
        h, g = self.paramSurfaceUi1.value()
        # _, eps = self.paramSurfaceUi2.value()
        xy, yz, xz, xyz = self.attr(
            parameters=(h, g, self.epsSlider.value()),
            iterations=self.skipSlider.value() + self.iterSlider.value(),
            skip=self.skipSlider.value(),
        )
        self.attrXY.setTexture(xy)
        self.attrYZ.setTexture(yz)
        self.attrXZ.setTexture(xz)
        self.attrXYZ.setTexture(xyz)

    def drawParamSurface(self):
        self.paramSurfaceUi1.setImage(self.paramSurface1())


if __name__ == '__main__':
    Task1().run()
