from dynsys import allocateImage, SimpleApp, Image2D, vStack, createSlider
import pyopencl as cl
import numpy

from examples.common import *

systemSource = r"""
#define Fz(z) (8.592*(z) - 22*(z)*(z) + 14.408*(z)*(z)*(z))

real3 userFn(real3, real, real, real);
real3 userFn(real3 v, real h, real g, real eps) {
    #define STEP (real)(1e-3)
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

COMMON_SOURCE = R"""
#define real float
#define real2 float2
#define real3 float3
"""

COMPUTE_SOURCE = r"""

#define MAP_BOUNDS(bs) \
    ((bs).s0 + (get_global_id(0))*((bs).s1 - (bs).s0)/(get_global_size(0)))

kernel void computeBifurcationTree(
    const real3 startPoint,
    float par0, float par1, float par2,
    const int activeAxisIdx,
    const real2 hardBounds,
    const int activeParamIdx,
    const real2 paramBounds,
    const int2 iterConf,
    global real3* result, 
    global real2* resultMinMax
) {
    const float param = MAP_BOUNDS(paramBounds);
    
    if        (activeParamIdx == 0) {
        par0 = param;
    } else if (activeParamIdx == 1) {
        par1 = param;
    } else if (activeParamIdx == 2) {
        par2 = param;
    }

    float3 point = startPoint;
    float min_;
    if        (activeAxisIdx == 0) {
        min_ = point.s0;
    } else if (activeAxisIdx == 1) {
        min_ = point.s1;
    } else if (activeAxisIdx == 2) {
        min_ = point.s2;
    }
    float max_;
    if (activeAxisIdx == 0) {
        max_ = point.s0;
    } else if (activeAxisIdx == 1) {
        max_ = point.s1;
    } else if (activeAxisIdx == 2) {
        max_ = point.s2;
    }
    
    for (int i = 0; i < iterConf.s0; ++i) {
        point = userFn(point, par0, par1, par2);
    }

    result += get_global_id(0) * iterConf.s1;
    for (int i = 0; i < iterConf.s1; ++i) {
        point = userFn(point, par0, par1, par2);
        
        if        (activeAxisIdx == 0) {
            if (point.s0 < min_ && point.s0 > hardBounds.s0) min_ = point.s0;
            if (point.s0 > max_ && point.s0 < hardBounds.s1) max_ = point.s0;
        } else if (activeAxisIdx == 1) {
            if (point.s1 < min_ && point.s1 > hardBounds.s0) min_ = point.s1;
            if (point.s1 > max_ && point.s1 < hardBounds.s1) max_ = point.s1;
        } else if (activeAxisIdx == 2) {
            if (point.s2 < min_ && point.s2 > hardBounds.s0) min_ = point.s2;
            if (point.s2 > max_ && point.s2 < hardBounds.s1) max_ = point.s2;
        }
        
        result[i] = point;
    }

    resultMinMax[get_global_id(0)] = (float2)(min_, max_); // save min/max
}
 
"""

DRAW_SOURCE = r"""

#define TREE_COLOR (float4)(0, 0, 0, 1.0)

#define TRANSLATE_BACK_1D(v, bs, size) \
    (((v) - (bs).s0) / ((bs).s1 - (bs).s0) * (size))

kernel void drawBifurcationTree(
    const real sliceValue,
    const int activeAxisIdx,
    const global float3* samples,
    const int iterations,
    const real2 bounds, 
    const real height,
    write_only image2d_t result
) {
    const int id = get_global_id(0);
    
    samples += id * iterations;
    
    for (int i = 1; i < iterations; ++i) {
        real v1, v2;
        if (activeAxisIdx == 0) {
            v1 = samples[i - 1].s0;
            v2 = samples[i].s0;
        } else if (activeAxisIdx == 1) {
            v1 = samples[i - 1].s1;
            v2 = samples[i].s1;
        } else if (activeAxisIdx == 2) {
            v1 = samples[i - 1].s2;
            v2 = samples[i].s2;
        }
        
        if (v2 < sliceValue && v1 >= sliceValue) {
            int2 coord = (int2)(id, height - TRANSLATE_BACK_1D(v2, bounds, height));
            write_imagef(result, coord, TREE_COLOR);
        }
    }
}

"""


class BifurcationTree:

    def __init__(self, ctx, queue, imageShape, mapFunctionSource, varCount, paramCount):
        self.ctx, self.queue = ctx, queue
        self.imageShape = imageShape
        sources = (
            COMMON_SOURCE, mapFunctionSource, COMPUTE_SOURCE, DRAW_SOURCE
        )
        self.varCount = varCount
        self.paramCount = paramCount
        self.hostImage, self.deviceImage = allocateImage(ctx, self.imageShape)
        self.program = cl.Program(ctx, "\n".join(sources)).build()

    def clear(self, readBack=False, color=(1.0, 1.0, 1.0, 1.0)):
        cl.enqueue_fill_image(
            self.queue, self.deviceImage,
            color=numpy.array(color, dtype=numpy.float32),
            origin=(0,)*len(self.imageShape), region=self.imageShape
        )
        if readBack:
            self.readFromDevice()

    def readFromDevice(self):
        cl.enqueue_copy(
            self.queue, self.hostImage, self.deviceImage,
            origin=(0,)*len(self.imageShape), region=self.imageShape
        )
        return self.hostImage

    def __call__(self,
                 sliceValue: float,
                 startPoint: tuple,
                 varIndex: int,
                 paramIndex: int,
                 paramRange: tuple,
                 otherParams: tuple,
                 iterations: int,
                 skip: int = 0,
                 maxAllowedValue: float = 1000.0):
        width = self.imageShape[0]

        resultDevice = cl.Buffer(
            self.ctx, cl.mem_flags.READ_WRITE, size=iterations * width * 4
        )
        resultMinMaxDevice = cl.Buffer(
            self.ctx, cl.mem_flags.WRITE_ONLY, size=width * 2 * 4
        )

        params = tuple(numpy.float32(par)
                       for par in (*otherParams[:paramIndex], numpy.nan, *otherParams[paramIndex+1:]))

        self.program.computeBifurcationTree(
            self.queue, (self.imageShape[0],), None,
            numpy.array(startPoint, dtype=numpy.float32), *params,
            numpy.int32(varIndex),
            numpy.array((-maxAllowedValue, maxAllowedValue), dtype=numpy.float32),
            numpy.int32(paramIndex),
            numpy.array(paramRange, dtype=numpy.float32),
            numpy.array((skip, iterations), dtype=numpy.int32),
            resultDevice, resultMinMaxDevice
        )

        resultMinMaxHost = numpy.empty((self.imageShape[0] * 2,), dtype=numpy.float32)

        cl.enqueue_copy(self.queue, resultMinMaxHost, resultMinMaxDevice)

        minMax = min(resultMinMaxHost), max(resultMinMaxHost)

        self.clear()

        self.program.drawBifurcationTree(
            self.queue, (self.imageShape[0],), None,
            numpy.float32(sliceValue),
            numpy.int32(varIndex),
            resultDevice,
            numpy.int32(iterations),
            numpy.array(minMax, dtype=numpy.float32),
            numpy.float32(self.imageShape[1]),
            self.deviceImage
        )

        return self.readFromDevice()


class Task3(SimpleApp):

    def __init__(self):
        super().__init__("Task 3")

        self.bifTree = BifurcationTree(
            self.ctx, self.queue, (512, 512), systemSource, 3, 3
        )
        self.bifTreeWidget = Image2D()
        self.slider1, self.slider1Ui = createSlider(
            "real", (-10, 10),
            labelPosition="top",
            withLabel="x = {}",
            withValue=-.5
        )
        self.slider2, self.slider2Ui = createSlider(
            "real", (-10, 10),
            labelPosition="top",
            withLabel="y = {}",
            withValue=3
        )
        self.slider3, self.slider3Ui = createSlider(
            "real", (-10, 10),
            labelPosition="top",
            withLabel="z = {}",
            withValue=8
        )

        self.sliderI, self.sliderIUi = createSlider(
            "int", (0, 10000),
            withLabel="skip = {}",
            withValue=0,
            labelPosition="top"
        )

        self.slider1.valueChanged.connect(self.drawBifTree)
        self.slider2.valueChanged.connect(self.drawBifTree)
        self.slider3.valueChanged.connect(self.drawBifTree)
        self.sliderI.valueChanged.connect(self.drawBifTree)

        self.setLayout(
            vStack(self.bifTreeWidget,
                   self.slider1Ui,
                   self.slider2Ui,
                   self.slider3Ui,
                   self.sliderIUi
                   )
        )

        self.drawBifTree()

    def drawBifTree(self, *_):
        self.bifTreeWidget.setTexture(self.bifTree(
            sliceValue=.5,
            startPoint=(self.slider1.value(), self.slider2.value(), self.slider3.value()),
            varIndex=0,
            paramIndex=1,
            paramRange=(0.75, 0.95),
            otherParams=(0.092, None, 0.2),
            iterations=1024,
            skip=self.sliderI.value()
        ))


if __name__ == '__main__':
    Task3().run()
