from dynsys import Bounds, SimpleApp, hStack, vStack, Qt, FLOAT, QLabel, \
    ParameterizedImageWidget, Image2D, Image3D, RealSlider, createSlider
from dynsys.PhasePlotV2 import PhasePlot, OutputConfig

from dynsys.ParameterMap import ParameterMap
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
iterations = skip + 256

systemSource = r"""

#define Fz(z) (8.592*(z) - 22*(z)*(z) + 14.408*(z)*(z)*(z))

real3 userFn(real3, real, real, real);
real3 userFn(real3 v, real h, real g, real eps) {
    #define STEP (real)(5e-3)
    real3 p = (real3)(
        2.0f*h*v.x + v.y - g*v.z, 
        -v.x,
        (v.x - Fz(v.z)) / eps
    );
    return v + STEP*p;
}
#define DYNAMIC_COLOR
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

        # self.paramSurface2, self.paramSurfaceUi2 = self.makeParameterSurface(
        #     source=parameterSurfaceSource,
        #     spaceShape=parameterMapBoundsHEps,
        #     withUi=True,
        #     uiNames=("h", "eps"),
        # )

        self.epsSlider, self.epsSliderWidget = createSlider("real", epsBounds,
                                      withLabel="Epsilon = {}",
                                      labelPosition="top",
                                      withValue=0.2,
                                      connectTo=self.drawAttractor)

        self.attr = PhasePlot(self.ctx, self.queue, (6, 6, 6), phaseBounds,
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
        # self.paramSurfaceUi2.valueChanged.connect(self.drawAttractor)

        # def update1When2Changed(val):
        #     _, g = self.paramSurfaceUi1.value()
        #     self.paramSurfaceUi1.setValue((val[0], g))
        #
        # def update2When1Changed(val):
        #     _, eps = self.paramSurfaceUi2.value()
        #     self.paramSurfaceUi2.setValue((val[0], eps))

        # self.paramSurfaceUi1.valueChanged.connect(update2When1Changed)
        # self.paramSurfaceUi2.valueChanged.connect(update1When2Changed)
        #

        # self.paramSurfaceUi2.setValue((0.07, 0.2))

        self.setLayout(
            vStack(
                hStack(
                    vStack(QLabel("X-Y"), self.attrXY),
                    vStack(QLabel("Y-Z"), self.attrYZ),
                    vStack(QLabel("X-Z"), self.attrXZ),
                ),
                hStack(
                    vStack(
                        self.paramSurfaceUi1, self.epsSliderWidget
                    ),
                    self.attrXYZ
                )
            )
        )
        self.attrXYZ.setFixedSize(512, 420)

        self.paramSurfaceUi1.setValue((0.07, 0.85))
        self.drawParamSurface()
        self.drawAttractor()

    def drawAttractor(self, *_):
        h, g = self.paramSurfaceUi1.value()
        # _, eps = self.paramSurfaceUi2.value()
        xy, yz, xz, xyz = self.attr(
            parameters=(h, g, self.epsSlider.value()),
            iterations=iterations,
            skip=skip,
        )
        self.attrXY.setTexture(xy)
        self.attrYZ.setTexture(yz)
        self.attrXZ.setTexture(xz)
        self.attrXYZ.setTexture(xyz)

    def drawParamSurface(self):
        self.paramSurfaceUi1.setImage(self.paramSurface1())
        # self.paramSurfaceUi2.setImage(self.paramSurface2())


if __name__ == '__main__':
    Task1().run()
