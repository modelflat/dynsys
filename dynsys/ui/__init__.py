from .image_widgets import Image2D, Image3D
from .slider_widgets import createSlider, RealSlider, IntegerSlider

import numpy

from PyQt5.Qt import QVBoxLayout, QHBoxLayout, QLayout, QColor, QWidget, QLabel
from PyQt5.QtCore import Qt, pyqtSignal as Signal


def vStack(*args):
    l = QVBoxLayout()
    for a in args:
        if isinstance(a, QLayout):
            l.addLayout(a)
        else:
            l.addWidget(a)
    return l


def hStack(*args):
    l = QHBoxLayout()
    for a in args:
        if isinstance(a, QLayout):
            l.addLayout(a)
        else:
            l.addWidget(a)
    return l


class ParameterizedImageWidget(QWidget):

    selectionChanged = Signal(tuple, tuple)

    def __init__(self, bounds: tuple,
                 names: tuple = ("x", "y"), shape: tuple = (True, True),
                 targetColor: QColor = Qt.red):
        super().__init__()
        if len(bounds) != 4:
            raise NotImplementedError("Only 2-D parameterized images are supported")
        self._bounds = bounds
        self._names = names
        self._shape = shape
        self._positionLabel = QLabel()
        self._imageWidget = Image2D(targetColor=targetColor, targetShape=shape, spaceShape=bounds)

        self._imageWidget.selectionChanged.connect(self.selectionChanged)
        self._imageWidget.selectionChanged.connect(self.updatePositionLabel)

        self.setLayout(vStack(
            self._imageWidget,
            self._positionLabel
        ))

    def updatePositionLabel(self, value, buttons):
        self._positionLabel.setText("  |  ".join(
            filter(lambda x: x is not None,
                   (None if sh is None else "{} = {}".format(nm, vl) for sh, nm, vl in zip(self._shape, self._names, value)))
        ))

    def setImage(self, image: numpy.ndarray):
        self._imageWidget.setTexture(image)

    def value(self):
        return self._imageWidget.targetReal()

    def setValue(self, targetValue: tuple):
        self._imageWidget.setTargetReal(targetValue)
        self.updatePositionLabel(targetValue, (False, False))


