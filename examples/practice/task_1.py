from dynsys import SimpleApp
import pyopencl as cl
import numpy

USER_SOURCE = """

#define STEP (1e-2f)
private float3 fn(float3 v, float a, float b, float r) {
    return v + STEP * (float3)(
        v.x,
        v.y,
        v.z
    );
}

"""

REALLY_COMMON_SOURCE = r"""
// get_global_linear_id available only in CL 2+
#if (__OPENCL_VERSION__ > CL_VERSION_1_2)
#define get_global_linear_id() \
    ((get_global_id(2) - get_global_offset(2)) * get_global_size(1) * get_global_size(0)) + \
    ((get_global_id(1) - get_global_offset(1)) * get_global_size (0)) + \
    (get_global_id(0) - get_global_offset(0))
#endif

#if (__OPENCL_VERSION__ >= CL_VERSION_2_0)
#define UNROLL_LOOP(count) __attribute__((opencl_unroll_hint(count)))
#else
#define UNROLL_LOOP(count)
#endif

"""

SOURCE = r"""

//#define PARAMETER_COUNT 3
//#define USE_NATURAL_GRID

#ifndef UNROLL_HINT_SKIP
#define UNROLL_HINT_SKIP 8
#endif

#ifndef UNROLL_HINT_ITER
#define UNROLL_HINT_ITER 8
#endif

// this should be generated
#define PARAMETERS_FROM_BUFFER(buffer) const float3 _params = (float3)( _params[0], _params[1], _params[2] ) 
#define PARAMETERS _params.s0, _params.s1, _params.s2

// this should be moved out
#define MAP_ID_TO_VALUE(bounds) \
    (float3)( \
        (bounds).s0 + get_global_id(0)*((bounds).s1 - (bounds).s0)/get_global_size(0), \
        (bounds).s2 + get_global_id(1)*((bounds).s3 - (bounds).s2)/get_global_size(1), \
        (bounds).s4 + get_global_id(2)*((bounds).s5 - (bounds).s4)/get_global_size(2))
    
#define BOUNDS_TYPE float8
#define POINT_TYPE float3

#define USER_SYSTEM fn

kernel void computeTrajectories(
#ifndef USE_NATURAL_GRID
    global const float3* startingPoints,
#endif 
    global const float* parameters,
    const BOUNDS_TYPE bounds,
    const ulong2 iterConf,
    global POINT_TYPE* buffer
) {
    buffer += get_global_linear_id() * iterConf.s1;

#ifdef USE_NATURAL_GRID
    POINT_TYPE point = MAP_ID_TO_VALUE(bounds);
#else
    POINT_TYPE point = startingPoints[get_global_linear_id()];
#endif

    PARAMETERS_FROM_BUFFER(parameters);
    
    UNROLL_LOOP(UNROLL_HINT_SKIP)
    for (size_t i = 0; i < iterConf.s0; ++i) {
        point = USER_SYSTEM(point, PARAMETERS);
    }
    
    UNROLL_LOOP(UNROLL_HINT_ITER)
    for (size_t i = 0; i < iterConf.s1; ++i) {
        point = USER_SYSTEM(point, PARAMETERS);
        buffer[i] = point;
    }
}

"""


PHASE_SOURCE = r"""

#define CONVERT_XY(point, bounds, image) \
    convert_int2_rtz((float2)( \
            ((point).x - (bounds).s0) / ((bounds).s1 - (bounds).s0)*get_image_width(image), \
            ((point).y - (bounds).s2) / ((bounds).s3 - (bounds).s2)*get_image_height(image) ))

#define CONVERT_YZ(point, bounds, image) \
    convert_int2_rtz((float2)( \
            ((point).y - (bounds).s2) / ((bounds).s3 - (bounds).s2)*get_image_width(image), \
            ((point).z - (bounds).s4) / ((bounds).s5 - (bounds).s4)*get_image_height(image) ))

#define CONVERT_XZ(point, bounds, image) \
    convert_int2_rtz((float2)( \
            ((point).x - (bounds).s0) / ((bounds).s1 - (bounds).s0)*get_image_width(image), \
            ((point).z - (bounds).s4) / ((bounds).s5 - (bounds).s4)*get_image_height(image) ))

#define CONVERT_XYZ(point, bounds, image) \
    (int4)convert_int3_rtz((float3)( \
            ((point).x - (bounds).s0) / ((bounds).s1 - (bounds).s0)*get_image_width (image), \
            ((point).y - (bounds).s2) / ((bounds).s3 - (bounds).s2)*get_image_height(image), \
            ((point).z - (bounds).s4) / ((bounds).s5 - (bounds).s4)*get_image_depth (image) ))


float3 hsv2rgb(float3 hsv) {
    const float c = hsv.y * hsv.z;
    const float x = c * (1 - fabs(fmod( hsv.x / 60, 2 ) - 1));
    float3 rgb;
    if      (0 <= hsv.x && hsv.x < 60) {
        rgb = (float3)(c, x, 0);
    } else if (60 <= hsv.x && hsv.x < 120) {
        rgb = (float3)(x, c, 0);
    } else if (120 <= hsv.x && hsv.x < 180) {
        rgb = (float3)(0, c, x);
    } else if (180 <= hsv.x && hsv.x < 240) {
        rgb = (float3)(0, x, c);
    } else if (240 <= hsv.x && hsv.x < 300) {
        rgb = (float3)(x, 0, c);
    } else {
        rgb = (float3)(c, 0, x);
    }
    return (rgb + (hsv.z - c));
}

float4 colorForPoint(int maxIter, int i, float3 point) {
    return hsv2rgb((float)i / maxIter * 240.0, 0, 0);
}

kernel void renderTrajectories(
    const global POINT_TYPE* buffer,
    const BOUNDS_TYPE bounds,
    const int iterations,
    const char8 conf,
    write_only image2d_t imageXY,
    write_only image2d_t imageYZ,
    write_inly image2d_t imageXZ,
    write_only image3d_t imageXYZ
) {
    float3 point;
    float4 color;
    buffer += get_global_id(0) * iterations;    
    for (size_t i = 0; i < iterations; ++i) {
        point = buffer[i];
        color = colorForPoint(iterations, i, point);
        if (conf.s0) {
            write_imagef(imageXY, CONVERT_XY(point, bounds, imageXY), color);
        }
        if (conf.s1) {
            write_imagef(imageYZ, CONVERT_YZ(point, bounds, imageYZ), color);
        }
        if (conf.s2) {
            write_imagef(imageXZ, CONVERT_XZ(point, bounds, imageXZ), color);
        }
        if (conf.s3) {
            write_imagef(imageXYZ, CONVERT_XYZ(point, bounds, imageXYZ), color);
        }
    }
}
    

"""


def wrapRunConf(ctx: cl.Context, skip: int, iteration: int):
    bits = set(dev.get_info(cl.device_info.ADDRESS_BITS) for dev in ctx.get_info(cl.context_info.DEVICES))
    if 32 in bits and 64 in bits:
        raise ValueError("Context include both 32 and 64 bit devices, do not know how to wrap size_t")
    if 32 in bits:
        return numpy.array((skip, iteration), dtype=numpy.int32)
    if 64 in bits:
        return numpy.array((skip, iteration), dtype=numpy.int64)


def wrapParameters(ctx, parameters, numpyType):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                     hostbuf=numpy.array(parameters, dtype=numpyType))


def wrapBuffer(ctx, numpyBuffer):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=numpyBuffer)


def wrapBounds(bounds, numpyType):
    if len(bounds) == 6:
        return numpy.array((*bounds, 0, 0), dtype=numpyType)
    elif len(bounds) == 4:
        return numpy.array(bounds, dtype=numpyType)
    raise NotImplementedError()


class TrajectoryEvolution:

    def __init__(self, ctx, queue, spaceShape, userFnSource, parameterCount,
                 useNaturalGrid=True, skipUnrollCount=8, iterUnrollCount=8,
                 dataRealType = numpy.float32
                 ):
        self.ctx, self.queue = ctx, queue
        self.spaceShape = spaceShape
        self.dim = len(spaceShape) // 2
        self.paramCount = parameterCount
        self.useGrid = useNaturalGrid
        self.program = cl.Program(ctx, "\n".join([REALLY_COMMON_SOURCE, userFnSource, SOURCE])) \
            .build(options=["-DPARAMETER_COUNT={}".format(parameterCount),
                            "-DUSE_NATURAL_GRID" if useNaturalGrid else "",
                            "-DUNROLL_HINT_ITER={}".format(iterUnrollCount),
                            "-DUNROLL_HINT_SKIP={}".format(skipUnrollCount),
                            "-cl-std=2.0"
                            ])
        self.realType = dataRealType

    def __call__(self, parameters, skip, iterations, gridShape: tuple = None, startingPoints = None):
        if len(parameters) != self.paramCount:
            raise ValueError("Invalid number of parameters: {}; required {}".format(len(parameters), self.paramCount))

        if self.useGrid:
            if gridShape is None:
                raise ValueError("gridShape is not given but useNaturalGrid is set")
        else:
            if startingPoints is None:
                raise ValueError("startingPoints is not given but useNaturalGrid is not set")

        if gridShape is not None and startingPoints is not None:
            raise ValueError("gridShape and startingPoints are mutually exclusive")

        if gridShape is not None and self.dim != len(gridShape):
            raise ValueError("gridShape is not of dimension {}".format(self.dim))

        runShape = (startingPoints.shape[0],) if gridShape is None else gridShape

        paramBuf = wrapParameters(self.ctx, parameters, self.realType)
        boundsBuf = wrapBounds(self.spaceShape, self.realType)
        runConfBuf = wrapRunConf(self.ctx, skip, iterations)
        outBuf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                           size=self.realType().nbytes*iterations*numpy.prod(runShape)*self.dim)

        if startingPoints is None:
            self.program.computeTrajectories(
                self.queue, runShape, None, paramBuf, boundsBuf, runConfBuf, outBuf
            )
        else:
            pointsBuf = wrapBuffer(self.ctx, startingPoints)
            self.program.computeTrajectories(
                self.queue, runShape, None, pointsBuf, paramBuf, boundsBuf, runConfBuf, outBuf
            )

        return outBuf


class PhasePlot:

    def __init__(self, ctx, queue, spaceShape, parameterCount):
        self.ctx, self.queue = ctx, queue
        self.spaceShape = spaceShape
        self.paramCount = parameterCount
        self.program = cl.Program(ctx, PHASE_SOURCE).build(
            options=["-cl-std=2.0"]
        )

    def __call__(self, evolutionResults: cl.Buffer, resultsNPoints, resultsNIters):
        self.program.renderTrajectories(
            self.queue, (resultsNPoints,), None,
            
        )




class Test(SimpleApp):

    def __init__(self):
        super().__init__("Test")

        self.test = TrajectoryEvolution(self.ctx, self.queue, (-1, 1, -1, 1, -1, 1), USER_SOURCE, 3)

        self.test((1, 1, 1), 256, 256, gridShape=(20, 20, 20))

if __name__ == '__main__':
    Test().run()
