from dynsys import allocateImage, SimpleApp, Image2D, vStack, createSlider
import pyopencl as cl
import numpy

from examples.common import *


systemSource = r"""
#define Fz(z) (8.592*(z) - 22*(z)*(z) + 14.408*(z)*(z)*(z))

real3 userFn(real3, real, real, real);
real3 userFn(real3 v, real a, real b, real r) {
    #define STEP (real)(1e-3)
    real3 p = (real3)(
        a*(v.y - v.x), 
        r*v.x - v.y - v.x*v.z,
        -b*v.z + v.y*v.x
    );
    return v + STEP*p;
}

"""

COMMON_SOURCE = R"""
#define real float
#define real3 float3
"""


COMPUTE_SOURCE = R"""

#define ID_1D get_global_id(0)

#define SIZE_1D get_global_size(0)

#define TRANSLATE_1D(id, size, bs) ((bs).s0 + (id)*((bs).s1 - (bs).s0)/(size))

kernel void computeBifurcationTree(
    const float3 startPoint,
    float par0, float par1, float par2,
    const int activeAxisIdx,
    const int activeParamIdx,
    const float2 bounds,
    const float maxAllowedValue,
    const int skip,
    const int iterations,
    global float* result, 
    global float2* resultMinMax
) {
    const int id = ID_1D;
    const float param = TRANSLATE_1D(id, SIZE_1D, bounds);

    result += id * iterations;
    
    if        (activeParamIdx == 0) {
        par0 = param;
    } else if (activeParamIdx == 1) {
        par1 = param;
    } else if (activeParamIdx == 2) {
        par2 = param;
    }

    float3 x = startPoint;
    
    float min_;
    if        (activeAxisIdx == 0) {
        min_ = x.s0;
    } else if (activeAxisIdx == 1) {
        min_ = x.s1;
    } else if (activeAxisIdx == 2) {
        min_ = x.s2;
    } 
    
    float max_;
    if (activeAxisIdx == 0) {
        max_ = x.s0;
    } else if (activeAxisIdx == 1) {
        max_ = x.s1;
    } else if (activeAxisIdx == 2) {
        max_ = x.s2;
    }
    
    for (int i = 0; i < skip; ++i) {
        x = userFn(x, par0, par1, par2);
    }

    for (int i = 0; i < iterations; ++i) {
        x = userFn(x, par0, par1, par2);
        
        if (activeAxisIdx == 0) {
            const float v = x.s0;
            if (v < min_ && v > -maxAllowedValue) 
                min_ = v;
            if (v > max_ && v <  maxAllowedValue) 
                max_ = v;
        } else if (activeAxisIdx == 1) {
            const float v = x.s1;
            if (v < min_ && v > -maxAllowedValue) 
                min_ = v;
            if (v > max_ && v <  maxAllowedValue) 
                max_ = v;
        } else if (activeAxisIdx == 2) {
            const float v = x.s2;
            if (v < min_ && v > -maxAllowedValue) 
                min_ = v;
            if (v > max_ && v <  maxAllowedValue) 
                max_ = v;
        }
        
        if (activeAxisIdx == 0) {
            result[i] = clamp(x.s0, -maxAllowedValue, maxAllowedValue);
        } else if (activeAxisIdx == 1) {
            result[i] = clamp(x.s1, -maxAllowedValue, maxAllowedValue);
        } else if (activeAxisIdx == 2) {
            result[i] = clamp(x.s2, -maxAllowedValue, maxAllowedValue);
        }
    }

    resultMinMax[id] = (float2)(min_, max_); // save min/max
}
 
"""


DRAW_SOURCE = r"""

#define TREE_COLOR (float4)(0, 0, 0, 1.0)

#define TRANSLATE_BACK_1D(v, bs, size) \
    (((v) - (bs).s0) / ((bs).s1 - (bs).s0) * (size))

kernel void drawBifurcationTree(
    const float sliceX,
    const global float* samples,
    const int iterations,
    const float2 bounds, 
    const float height,
    write_only image2d_t result
) {
    const int id = get_global_id(0);
    
    samples += id * iterations;
    
    for (int i = 1; i < iterations; ++i) {
        if (samples[i - 1] >= sliceX && samples[i] < sliceX) {
            int2 coord = (int2)(id, height - TRANSLATE_BACK_1D(samples[i], bounds, height));
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
                 sliceX: float,
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
            numpy.int32(paramIndex),
            numpy.array(paramRange, dtype=numpy.float32),
            numpy.float32(maxAllowedValue),
            numpy.int32(skip),
            numpy.int32(iterations),
            resultDevice,
            resultMinMaxDevice
        )

        resultMinMaxHost = numpy.empty((self.imageShape[0]*2,), dtype=numpy.float32)

        cl.enqueue_copy(self.queue, resultMinMaxHost, resultMinMaxDevice)

        minMax = min(resultMinMaxHost), max(resultMinMaxHost)

        self.clear()

        self.program.drawBifurcationTree(
            self.queue, (self.imageShape[0],), None,
            numpy.float32(sliceX),
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
            self.ctx, self.queue, (768, 512), fn_Ressler, 3, 3
        )
        self.bifTreeWidget = Image2D()

        self.sliderI, self.sliderIUi = createSlider(
            "int", (0, 10000),
            withLabel="skip = {}",
            withValue=0,
            labelPosition="top"
        )

        self.sliderI.valueChanged.connect(self.drawBifTree)

        self.setLayout(
            vStack(self.bifTreeWidget, self.sliderIUi)
        )

        self.drawBifTree()

    def drawBifTree(self, *_):
        self.bifTreeWidget.setTexture(self.bifTree(
            sliceX=.5,
            startPoint=(1, 1, 1),
            varIndex=0,
            paramIndex=2,
            paramRange=(0, 150),
            otherParams=(10, 8/3, 2.5),
            iterations=8192, skip=self.sliderI.value(), maxAllowedValue=10
        ))


if __name__ == '__main__':
    Task3().run()
