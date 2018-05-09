import pyopencl as cl
from PyQt4 import QtGui
import numpy as np

def allocate_image(ctx, w, h) :
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    return cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))


def pixmap_from_raw_image(img: np.ndarray):
    image = QtGui.QImage(img.data, img.shape[0], img.shape[1], QtGui.QImage.Format_ARGB32)
    pixmap = QtGui.QPixmap()
    pixmap.convertFromImage(image)
    return pixmap


def type():
    return np.float64

def real(arg):
    return type()(arg)

def typesize():
    return 8


def translate(x, d, a, b):
    return a + (b - a)*x / d
