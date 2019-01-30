import sys

import numpy
import os
import pyopencl as cl
from PyQt5.QtWidgets import QCheckBox

from dynsys import SimpleApp, ComputedImage, FLOAT, ParameterizedImageWidget, vStack, createSlider, hStack

spaceShape = (-1.5, 1.5, -1.5, 1.5)
hBounds = (-2, 2)
alphaBounds = (0, 1)


paramSurfaceMap = r"""

float3 userFn(real2 v);
float3 userFn(real2 v) {
    real h = v.x;
    real alpha = v.y;
    if (h > 0 && alpha > 0.5) {
        if (get_global_id(0) % 2 == 0 && get_global_id(1) % 2 == 0) {
            return 0.0f;
        }
    }
    if (fabs(alpha - 1) < 0.002 && h < 0) {
        return (float3)(1, 0, 0);
    }
    if (length(v - (real2)(1.0, 0.0)) < 0.01) {
        return (float3)(1, 0, 0);
    }
    if (length(v - (real2)(-1.0, 0.0)) < 0.01) {
        return (float3)(1, 0, 0);
    }
    return 1.0f;
}

"""


SCRIPT_DIR = os.path.abspath(sys.path[0])
print(SCRIPT_DIR)


def readFile(path):
    with open(path) as file:
        return file.read()


class IFSFractal(ComputedImage):

    def __init__(self, ctx, queue, imageShape, spaceShape, fractalSource, options=[], staticColor=(0.0, 0.0, 0.0, 1.0)):
        super().__init__(ctx, queue, imageShape, spaceShape,
                         fractalSource,
                         options=[*options,
                                  "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])],
                         typeConfig=FLOAT)
        self.staticColor = staticColor

    def __call__(self, alpha: float, h: float, c: complex, pointCount: int, iterCount: int, skip: int):
        colorBuf = cl.Buffer(self.ctx, flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=numpy.array(self.staticColor, dtype=numpy.float32))

        self.clear()

        self.program.newton_fractal(
            self.queue, (pointCount,), None,
            numpy.array(self.spaceShape, dtype=numpy.float64),
            numpy.array((c.real, c.imag), dtype=numpy.float64),
            numpy.float64(alpha),
            numpy.float64(h),
            numpy.int32(iterCount),
            numpy.int32(skip),
            numpy.random.randint(low=0, high=0xFFFF_FFFF_FFFF_FFFF, dtype=numpy.uint64),
            self.deviceImage
        )

        return self.readFromDevice()


class CourseWork(SimpleApp):

    def __init__(self):
        super().__init__("Coursework")

        self.ifsf = IFSFractal(self.ctx, self.queue,
                               imageShape=(512, 512),
                               spaceShape=spaceShape,
                               fractalSource=readFile(os.path.join(SCRIPT_DIR, "newton_fractal.cl")),
                               options=["-I{}".format(os.path.join(SCRIPT_DIR, "include")),
                                        "-cl-std=CL1.0",
                                        ]
                               )
        self.ifsfUi = ParameterizedImageWidget(spaceShape, names=("z_real", "z_imag"), shape=(True, True),
                                               textureShape=(512, 512))

        self.alphaHParamSurf, self.alphaHParamSurfUi = self.makeParameterSurface(
            paramSurfaceMap, spaceShape=(*hBounds, *alphaBounds),
            imageShape=(510, 510),
            uiNames=("h", "alpha"), uiShape=(True, True),
        )

        def setSlidersAndDraw(val):
            self.draw()
            self.hSlider.setValue(val[0])
            self.alphaSlider.setValue(val[1])

        self.alphaHParamSurfUi.valueChanged.connect(setSlidersAndDraw)

        def setHValue(h):
            _, alpha = self.alphaHParamSurfUi.value()
            self.alphaHParamSurfUi.setValue((h, alpha))
            self.draw()

        self.hSlider, self.hSliderUi = createSlider(
            "real", hBounds,
            withLabel="h = {:2.3f}",
            labelPosition="top",
            withValue=.5,
            connectTo=setHValue
        )

        def setAlphaValue(alpha):
            h, _ = self.alphaHParamSurfUi.value()
            self.alphaHParamSurfUi.setValue((h, alpha))
            self.draw()

        self.alphaSlider, self.alphaSliderUi = createSlider(
            "real", alphaBounds,
            withLabel="alpha = {:2.3f}",
            labelPosition="top",
            withValue=0.0,
            connectTo=setAlphaValue
        )

        self.setLayout(
            hStack(
                self.alphaHParamSurfUi,
                vStack(
                    self.ifsfUi,
                    self.alphaSliderUi,
                    self.hSliderUi,
                )
            )
        )
        self.alphaHParamSurfUi.setImage(self.alphaHParamSurf())
        self.draw()

    def draw(self, *_):
        h, alpha = self.alphaHParamSurfUi.value()
        self.ifsfUi.setImage(self.ifsf(
            alpha=alpha,
            h=h,
            c=complex(-0.5, 0.5),
            pointCount=32,
            iterCount=4096,
            skip=1024
        ))


if __name__ == '__main__':
    CourseWork().run()
