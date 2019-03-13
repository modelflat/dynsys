import os

import sys
from PyQt5.Qt import QPalette
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QLineEdit, QCheckBox, QPushButton

from dynsys import SimpleApp, ComputedImage, FLOAT, ParameterizedImageWidget, vStack, createSlider, hStack
from fbc.boxcount_original import *

spaceShape = (-1., 1., -1., 1.)
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


def random_seed():
    return numpy.random.randint(low=0, high=0xFFFF_FFFF_FFFF_FFFF, dtype=numpy.uint64),


def prepare_root_seq(ctx, root_seq):
    if root_seq is None:
        seq = numpy.empty((1,), dtype=numpy.int32)
        seq[0] = -1
    else:
        seq = numpy.array(root_seq, dtype=numpy.int32)

    seq_buf = cl.Buffer(ctx, flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=seq)
    return seq.size if root_seq is not None else 0, seq_buf


class IFSFractalParameterMap(ComputedImage):

    def __init__(self, ctx, queue, imageShape, spaceShape, fractalSource, options=[]):
        super().__init__(ctx, queue, imageShape, spaceShape,
                         fractalSource,
                         options=[*options, "-w",
                                  "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])],
                         typeConfig=FLOAT)
        self.points: cl.Buffer = None

    def compute_points(self, z0: complex, c: complex, skip: int, iter: int, tol: float, root_seq=None,
                       wait=False):
        reqd_size = 2 * iter * numpy.prod(self.imageShape)

        if self.points is None or self.points.size != reqd_size:
            self.points = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                    size=reqd_size * 4)

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.program.compute_points(
            self.queue, self.imageShape, (1, 1),
            # z0
            numpy.array((z0.real, z0.imag), dtype=numpy.float64),
            # c
            numpy.array((c.real, c.imag), dtype=numpy.float64),
            # bounds
            numpy.array(self.spaceShape, dtype=numpy.float64),
            # skip
            numpy.int32(skip),
            # iter
            numpy.int32(iter),
            # tol
            numpy.float32(tol),
            # seed
            numpy.float64(random_seed()),
            # seq size
            numpy.int32(seq_size),
            # seq
            seq,
            # result
            self.points
        )
        if wait:
            self.queue.finish()

    def display(self, period_range: tuple, num_points: int):
        color_scheme = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, size=4)

        self.program.draw_periods(
            self.queue, self.imageShape, (1, 1),
            numpy.int32(period_range[0]),
            numpy.int32(period_range[1]),
            numpy.int32(num_points),
            color_scheme,
            self.points,
            cl.LocalMemory(2 * 4 * num_points),
            self.deviceImage
        )

        self.queue.finish()


class IFSFractal(ComputedImage):

    def __init__(self, ctx, queue, imageShape, spaceShape, fractalSource, options=[], staticColor=(0.0, 0.0, 0.0, 1.0)):
        super().__init__(ctx, queue, imageShape, spaceShape,
                         fractalSource,
                         options=[*options, "-w",
                                  "-D_{}".format(numpy.random.randint(0, 2**64-1, size=1, dtype=numpy.uint64)[0])],
                         typeConfig=FLOAT)
        self.staticColor = staticColor

    def __call__(self, alpha: float, h: float, c: complex, grid_size: int, iterCount: int, skip: int,
                 root_seq=None, clear_image=True):

        if clear_image:
            self.clear()

        seq_size, seq = prepare_root_seq(self.ctx, root_seq)

        self.program.newton_fractal(
            self.queue, (grid_size, grid_size), None,
            numpy.int32(skip),
            numpy.int32(iterCount),

            numpy.array(self.spaceShape, dtype=numpy.float64),

            numpy.array((c.real, c.imag), dtype=numpy.float64),
            numpy.float64(h),
            numpy.float64(alpha),

            numpy.random.randint(low=0, high=0xFFFF_FFFF_FFFF_FFFF, dtype=numpy.uint64),

            numpy.int32(seq_size),
            seq,

            self.deviceImage
        )

        return self.readFromDevice()


class CourseWork(SimpleApp):

    def __init__(self):
        super().__init__("Coursework")

        source = readFile(os.path.join(SCRIPT_DIR, "newton_fractal.cl"))

        self.ifsf = IFSFractal(self.ctx, self.queue,
                               imageShape=(512, 512),
                               spaceShape=spaceShape,
                               fractalSource=source,
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

        self.param_map = IFSFractalParameterMap(self.ctx, self.queue,
                                                imageShape=(64, 64),
                                                spaceShape=(*hBounds, *alphaBounds),
                                                fractalSource=source,
                                                options=["-I{}".format(os.path.join(SCRIPT_DIR, "include")),
                                                         "-cl-std=CL1.0",
                                                         ]
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

        self.dLabel = QLabel("D = ???")
        self.rootSeqEdit = QLineEdit()
        self.rootSeqEditPalette = QPalette()
        self.rootSeqEditPalette.setColor(QPalette.Text, Qt.black)
        self.rootSeqEdit.setPalette(self.rootSeqEditPalette)

        self.genRandom = QPushButton("Generate random (input length)")
        self.genRandom.clicked.connect(self.genRandomSeqFn)
        self.randomSeq = None

        self.resetRandomSeq = QPushButton("Reset")
        self.resetRandomSeq.clicked.connect(self.resetRandomSeqFn)

        self.refreshButton = QPushButton("Refresh")
        self.refreshButton.clicked.connect(self.draw)

        self.shouldClear = QCheckBox("Clear image")
        self.shouldClear.setChecked(True)

        self.setLayout(
            hStack(
                vStack(
                    self.alphaHParamSurfUi,
                    self.dLabel,
                    hStack(self.genRandom, self.resetRandomSeq),
                    self.rootSeqEdit
                ),
                vStack(
                    self.ifsfUi,
                    hStack(self.refreshButton, self.shouldClear),
                    self.alphaSliderUi,
                    self.hSliderUi,
                )
            )
        )

        self.count = FastBoxCounting(self.ctx)

        self.recompute_param_map()

        self.alphaHParamSurfUi.setImage(self.alphaHParamSurf())
        self.draw()

    def resetRandomSeqFn(self, *_):
        self.randomSeq = None
        self.dLabel.setText("[-1]")

    def genRandomSeqFn(self, *_):
        try:
            rootSeqSize = int(self.rootSeqEdit.text())
            self.randomSeq = numpy.random.randint(0, 2 + 1, size=rootSeqSize, dtype=numpy.int32)
        except:
            pass

    def parseRootSequence(self):
        raw = self.rootSeqEdit.text()
        if self.randomSeq is not None:
            return self.randomSeq
        else:
            l = list(map(int, raw.split()))
            if len(l) == 0 or not all(map(lambda x: x <= 2, l)):
                return None
            else:
                return l

    def recompute_param_map(self, *_):

        n_iter = 64
        n_skip = 256

        import time
        print("Start computing parameter map")
        t = time.perf_counter()
        self.param_map.compute_points(
            z0=complex(0.0, 0.0),
            c=complex(0.0, 0.5),
            skip=n_skip,
            iter=n_iter,
            tol=0.005,
            root_seq=None,
            wait=True
        )
        t = time.perf_counter() - t
        print("Computed parameter map in {:.3f} s".format(t))
        print("Trying to draw")
        t = time.perf_counter()
        self.param_map.display(
            period_range=(2, 64),
            num_points=n_iter
        )
        t = time.perf_counter() - t
        print("Drawn in {:.3f} s".format(t))

    def draw(self, *_):
        h, alpha = self.alphaHParamSurfUi.value()

        try:
            seq = self.parseRootSequence()
        except:
            seq = None

        image = self.ifsf(
            alpha=alpha,
            h=h,
            c=complex(-0.0, 0.5),
            grid_size=2,
            iterCount=8192 << 2,
            skip=0,
            root_seq=seq,
            clear_image=self.shouldClear.isChecked()
        )
        # d = self.count.count_for_image(self.queue, (512, 512), self.ifsf.deviceImage)
        d = 1
        # self.dLabel.setText("D = {:6.3f}".format(d))
        # self.dLabel.setText("Root sequence: {}".format(str(rootSeq)))
        self.ifsfUi.setImage(image)


if __name__ == '__main__':
    CourseWork().run()
