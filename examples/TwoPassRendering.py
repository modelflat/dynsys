from dynsys import SimpleApp, ComputedImage, FLOAT, Image2D, vStack

import pyopencl as cl
import numpy

RANDOM = r"""
// source code:
// http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
uint MWC64X(uint2*);
uint MWC64X(uint2 *state) {
    enum{ A=4294883355U };
    uint x = (*state).x, c = (*state).y;  // Unpack the state
    uint res = x^c;                     // Calculate the result
    uint hi = mul_hi(x,A);              // Step the RNG
    x = x*A+c;
    c = hi+(x<c);
    *state = (uint2)(x,c);               // Pack the state back up
    return res;                        // Return the next result
}

// ! this rng does not take into account local id, but it can
// be mixed in one of seed values if needed
void init_state(ulong, uint2*);
void init_state(ulong seed, uint2* state) {
    int id = get_global_id(0) + 1;
    uint2 s = as_uint2(seed);
    (*state) = (uint2)(
        // create a mixture of id and two seeds
        (id + s.x & 0xFFFF) * s.y,
        (id ^ (s.y & 0xFFFF0000)) ^ s.x
    );
}

// retrieve random float in range [0.0; 1.0] (both inclusive)
float random(uint2*);
float random(uint2* state) {
    return ((float)MWC64X(state)) / (float)0xffffffff;
}
"""

SOURCE = RANDOM + r"""

float2 userFn(float2, float, float);
float2 userFn(float2 v, float a, float b) {
    return (float2)(
        1 - a*v.x*v.x + b*v.y,
        v.x
    );
}

float4 userColor(int);
float4 userColor(int packedData) {
    if (packedData == 0)
        return (float4)(0.0, 0.0, 0.0, 1.0);
    else
        return (float4)(1.0, 0.0, 0.0, 1.0);
}

#define ID_2D (int2)(get_global_id(0), get_global_id(1))

#define SIZE_2D (int2)(get_global_size(0), get_global_size(1))

#define TRANSLATE_BACK_2D(T, v, bs, size) \
    (T)(((v).x - (bs).s0)/((bs).s1 - (bs).s0)*(size).x, \
        ((v).y - (bs).s2)/((bs).s3 - (bs).s2)*(size).y )

/// ---

kernel void iterateFunctionSystem(
    const float4 bounds,
    const float4 starting,
    const float2 params,
    const int skip,
    const int iterations,
    global int* buffer
) {
    uint2 state;
    init_state(0x42, &state);
    
    float2 point = (float2)(
        starting.s0 + (starting.s1 - starting.s0) * random(&state),
        starting.s2 + (starting.s3 - starting.s2) * random(&state)
    );
    
    for (int i = 0; i < skip; ++i) {
        point = userFn(point, params.s0, params.s1);
    }
    
    for (int i = skip; i < iterations; ++i) {
        point = userFn(point, params.s0, params.s1);
        float2 coord = TRANSLATE_BACK_2D(float2, point, bounds, SIZE_2D);
        int2 coordInt = convert_int2_rtz(coord);
        
        buffer[coordInt.x * get_global_size(1) + coordInt.y] = 1;
    }
}


kernel void colorizeImage(
    global int* buffer, write_only image2d_t outImage
) {
    const int2 id = ID_2D;
    write_imagef(outImage, id, userColor(buffer[id.x * get_global_size(1) + id.y]));
}


"""


class IFS(ComputedImage):

    def __init__(self, ctx, queue, space):
        super().__init__(
            ctx, queue, (512, 512), space,
            SOURCE,
            typeConfig=FLOAT
        )
        self.buffer = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                                size=4 * self.imageShape[0] * self.imageShape[1])

    def __call__(self, params: tuple, startingLocation: tuple, skip: int, iterations: int):

        self.clear()

        self.program.iterateFunctionSystem(
            self.queue, self.imageShape, None,
            numpy.array(self.spaceShape, dtype=numpy.float32),
            numpy.array(startingLocation, dtype=numpy.float32),
            numpy.array(params, dtype=numpy.float32),
            numpy.int32(skip), numpy.int32(iterations),
            self.buffer
        )

        self.program.colorizeImage(
            self.queue, self.imageShape, None,
            self.buffer, self.deviceImage
        )

        return self.readFromDevice()


class Test(SimpleApp):

    def __init__(self):
        super(Test, self).__init__("test")
        self.ifs = IFS(self.ctx, self.queue, (-1.0, 1.0, -1.0, 1.0))
        self.imw = Image2D()
        self.imw.setTexture(self.ifs((1.5, 1), (0.5, 0.6, 0.5, 0.6),
                                     255, 256))

        self.setLayout(
            vStack(self.imw)
        )


if __name__ == '__main__':
    Test().run()
