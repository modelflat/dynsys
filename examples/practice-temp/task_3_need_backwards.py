import numpy
import pyopencl as cl

from dynsys import allocateImage, SimpleApp, Image2D, vStack, ParameterizedImageWidget, hStack
from dynsys.LCE import dummyOption


BOGDANOV = r"""

#define real double
#define real2 double2
#define real4 double4

#define userFn_SOURCE_f(x, y, u, eps) (real2)( \
    x + eps*(y + eps*((u - x)*y - x*x)), \
    y + eps*((u - x)*y - x*x) \
)

#define userFn_SOURCE_b(x, y, u, eps) (real2)( \
    x - eps*y, \
    (y + eps*(1 + (x - eps*y)*(x - eps*y))) / (1 + eps *(u - x + eps*y)) \
)

"""

SOURCE = r"""

kernel void evolve(
    const global real* startingPoints,
    const real p1, const real p2,
    const real4 bounds,
    const int iteration,
    const int isForward,
    global real2* endPoints,
    write_only image2d_t output
) {
    real2 point = vload2(get_global_id(0), startingPoints);
    const int skip = 0; //iteration / 4;
    for (int i = 0; i < iteration; ++i) {
        if (i == 0) {
            int2 coord = (int2)(
                (point.x - bounds.s0) / (bounds.s1 - bounds.s0) * get_image_width(output),
                (1 - (point.y - bounds.s2) / (bounds.s3 - bounds.s2)) * get_image_height(output)
            );
            if (coord.x >= 0 && coord.x < get_image_width(output) && coord.y >= 0 && coord.y < get_image_height(output)) {
                write_imagef(output, coord, (float4)(1, 0, 0, 1));
            }
        }
        
    
        if (isForward) {
            point = userFn_SOURCE_f(point.x, point.y, p1, p2);
        } else {
            point = userFn_SOURCE_b(point.x, point.y, p1, p2);
        }
        
        int2 coord = (int2)(
            (point.x - bounds.s0) / (bounds.s1 - bounds.s0) * get_image_width(output),
            (1 - (point.y - bounds.s2) / (bounds.s3 - bounds.s2)) * get_image_height(output)
        );
        if (i > skip && coord.x >= 0 && coord.x < get_image_width(output) && coord.y >= 0 && coord.y < get_image_height(output)) {
            if (isForward){
                write_imagef(output, coord, (float4)(1, 0, 0, 1));
            } else{
                write_imagef(output, coord, (float4)(0, 1, 0, 1));
            }
        }
    }
    endPoints[get_global_id(0)] = point;
}

"""


def generatePointsInEps(stablePoint: tuple, eps: tuple, pointCount: int, dtype=numpy.float64):
    return numpy.array((
        numpy.random.uniform(
            low=stablePoint[0] - eps[0],
            high=stablePoint[0] + eps[0],
            size=pointCount
        ),
        numpy.random.uniform(
            low=stablePoint[1] - eps[1],
            high=stablePoint[1] + eps[1],
            size=pointCount
        )
    ), dtype=dtype).T.copy()


class Homoclinic:

    def __init__(self, ctx, queue, spaceShape, imageShape=(512, 512)):
        self.ctx, self.queue = ctx, queue
        self.prg = cl.Program(self.ctx, "\n".join(
            (BOGDANOV, SOURCE,)
        )).build(options=[dummyOption()])
        self.spaceShape = spaceShape
        self.type = numpy.float64
        self.imageShape = imageShape
        self.hostImage, self.deviceImage = allocateImage(self.ctx, self.imageShape)

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

    def __call__(self, stablePoint, eps, pointCount, p1, p2, iterations):
        points = generatePointsInEps(stablePoint, eps, pointCount, self.type)
        print(points)
        pointsDev = cl.Buffer(self.ctx, cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, hostbuf=points)

        end = numpy.empty(points.shape, dtype=self.type)
        endDev = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=end.nbytes)

        self.clear()

        # forward
        self.prg.evolve(
            self.queue, (pointCount,), None,
            pointsDev,
            self.type(p1), self.type(p2),
            numpy.array(self.spaceShape, dtype=self.type),
            numpy.int32(iterations),
            numpy.int32(1),
            endDev, self.deviceImage
        )

        #backward
        self.prg.evolve(
            self.queue, (pointCount,), None,
            pointsDev,
            self.type(p1), self.type(p2),
            numpy.array(self.spaceShape, dtype=self.type),
            numpy.int32(iterations),
            numpy.int32(0),
            endDev, self.deviceImage
        )

        return self.readFromDevice()


class Test(SimpleApp):

    def __init__(self):
        super().__init__("123")
        self.homo = Homoclinic(self.ctx, self.queue, (-5, 5, -5, 5))

        self.sel = ParameterizedImageWidget(
            bounds=(-1, 1, -1, 1), names=("u", "eps"),
            shape=(512, 512)
        )
        self.sel._imageWidget.setTexture(numpy.empty((512, 512, 4), dtype=numpy.int32))
        self.sel.valueChanged.connect(self.draw)

        self.lab = Image2D()
        self.setLayout(
            hStack(self.sel, self.lab)
        )

        self.draw()


    def draw(self):
        u, eps = self.sel.value()
        self.lab.setTexture(
            self.homo((1, 0), (1e-3, 1e-3), 1024, u, eps, 1000)
        )


class Henon(SimpleApp):

    def __init__(self):
        super().__init__("123")
        self.homo = Homoclinic(self.ctx, self.queue, (-5, 5, -5, 5))

        self.sel = ParameterizedImageWidget(
            bounds=(-1, 1, -1, 1), names=("u", "eps"),
            shape=(512, 512)
        )
        self.sel._imageWidget.setTexture(numpy.empty((512, 512, 4), dtype=numpy.int32))
        self.sel.valueChanged.connect(self.draw)

        self.lab = Image2D()
        self.setLayout(
            hStack(self.sel, self.lab)
        )

        self.draw()


    def draw(self):
        u, eps = self.sel.value()
        self.lab.setTexture(
            self.homo((1, 0), (1e-3, 1e-3), 1024, u, eps, 1000)
        )


if __name__ == '__main__':
    Test().run()
