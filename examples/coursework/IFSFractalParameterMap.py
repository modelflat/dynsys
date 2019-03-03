import numpy
import pyopencl as cl
from dynsys import ComputedImage, FLOAT


class IFSFractalParameterMap(ComputedImage):

    def __init__(self, ctx, queue, imageShape, spaceShape, fractalSource, options=[]):
        super().__init__(ctx, queue, imageShape, spaceShape,
                         fractalSource,
                         options=[*options,
                                  "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])],
                         typeConfig=FLOAT)
        self.color_scheme = numpy.array((
            (0.0, 0.0, 0.0, 1.0), # inf or other
            (1.0, 0.0, 0.0, 1.0), # 1
            (0.0, 1.0, 0.0, 1.0), # 2
            (0.0, 0.0, 1.0, 1.0), # 3
            (1.0, 0.0, 1.0, 1.0), # 4
            (1.0, 1.0, 0.0, 1.0), # 5
            (0.0, 1.0, 1.0, 1.0), # 6
            (0.5, 0.0, 0.0, 1.0), # 7
            (0.0, 0.5, 0.5, 1.0)  # 8
        ), dtype=numpy.float32)
        self.color_scheme_device = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                             hostbuf=self.color_scheme)

    def __call__(self, periods: int, skip: int, tol: float, c: complex, init: complex,
                 color_scheme=None, root_seq=None, clear_image=True):

        if periods > 8 or color_scheme is not None:
            raise NotImplementedError("this Parameter Map currently does not support periods >= 8 due to color scheme limitations")

        if root_seq is None:
            rootSequence = numpy.empty((1,), dtype=numpy.int32)
            rootSequence[0] = -1
        else:
            rootSequence = numpy.array(root_seq, dtype=numpy.int32)

        rootSeqBuf = cl.Buffer(self.ctx, flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=rootSequence)

        temp_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                             size=2 * periods * 8 * numpy.prod(self.imageShape))

        if clear_image:
            self.clear()

        self.program.parameter_map(
            self.queue, self.imageShape, None,
            numpy.int32(skip),
            numpy.int32(periods),
            numpy.float64(tol),

            numpy.array((c.real, c.imag), dtype=numpy.float64),
            numpy.array((init.real, init.imag), dtype=numpy.float64),

            numpy.array(self.spaceShape, dtype=numpy.float64),

            numpy.random.randint(low=0, high=0xFFFF_FFFF_FFFF_FFFF, dtype=numpy.uint64),

            numpy.int32(rootSequence.size if root_seq is not None else 0),
            rootSeqBuf,

            self.color_scheme_device,

            temp_buf,

            self.deviceImage
        )

        return self.readFromDevice(), rootSequence


"""

        self.ifsfpm = IFSFractalParameterMap(self.ctx, self.queue,
                               imageShape=(512, 512),
                               spaceShape=(*hBounds, *alphaBounds),
                               fractalSource=readFile(os.path.join(SCRIPT_DIR, "newton_fractal.cl")),
                               options=["-I{}".format(os.path.join(SCRIPT_DIR, "include")),
                                        "-cl-std=CL1.0",
                                        ]
                               )
        self.ifsfpmUi = ParameterizedImageWidget(bounds=(*hBounds, *alphaBounds),
                                               names=("h", "alpha"),
                                               shape=(True, True),
                                               textureShape=(512, 512))
                                               

    def draw_parameter_map(self, *_):
        image, root_seq = self.ifsfpm(
            periods=8,
            skip=1,
            tol=0.0001,
            c=complex(-0.0, 0.5),
            init=complex(-0.5, 0.5),
            root_seq=None
        )
        self.ifsfpmUi.setImage(image)
        

"""