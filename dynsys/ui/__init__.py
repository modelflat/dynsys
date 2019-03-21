from .ImageWidgets import Image2D, Image3D
from .SliderWidgets import createSlider, RealSlider, IntegerSlider

import numpy

from PyQt5.Qt import QVBoxLayout, QHBoxLayout, QLayout, QColor, QWidget, QLabel
from PyQt5.QtCore import Qt, pyqtSignal as Signal


def vStack(*args, cm=(0, 0, 0, 0)):
    l = QVBoxLayout()
    l.setContentsMargins(*cm)
    l.setSpacing(0)
    for a in args:
        if isinstance(a, QLayout):
            l.addLayout(a)
        else:
            l.addWidget(a)
    return l


def hStack(*args, cm=(0, 0, 0, 0)):
    l = QHBoxLayout()
    l.setContentsMargins(*cm)
    l.setSpacing(0)
    for a in args:
        if isinstance(a, QLayout):
            l.addLayout(a)
        else:
            l.addWidget(a)
    return l


class ParameterizedImageWidget(QWidget):

    selectionChanged = Signal(tuple, tuple)

    valueChanged = Signal(tuple)

    def __init__(self, bounds: tuple, names: tuple, shape: tuple, targetColor: QColor = Qt.red, textureShape=(1,1)):
        super().__init__()
        # if len(bounds) != 4:
        #     raise NotImplementedError("Only 2-D parameterized images are supported")

        if names is None and shape is None:
            names = ("", "")
            shape = (True, True)

        if names is None:
            names = ("", "")

        if shape is None:
            shape = tuple(bool(el) if el is not None else False for el in names)

        self._bounds = bounds
        self._names = names
        self._shape = shape
        self._positionLabel = QLabel()
        self._imageWidget = Image2D(targetColor=targetColor, targetShape=shape, spaceShape=bounds, textureShape=textureShape)

        self._imageWidget.selectionChanged.connect(lambda val, _: self.valueChanged.emit(val))
        self._imageWidget.selectionChanged.connect(self.selectionChanged)
        self._imageWidget.selectionChanged.connect(self.updatePositionLabel)

        self.setLayout(vStack(
            self._imageWidget,
            self._positionLabel
        ))

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        # self.updatePositionLabel(self._imageWidget._target.)

    def updatePositionLabel(self, value, buttons):
        self._positionLabel.setText("  |  ".join(
            filter(lambda x: x is not None, (
                None if sh is None or vl is None else "{0} = {1:.4f}".format(nm, vl)
                for sh, nm, vl in zip(self._shape, self._names, value)
            ))
        ))

    def setImage(self, image: numpy.ndarray):
        self._imageWidget.setTexture(image)

    def value(self):
        return self._imageWidget.targetReal()

    def setValue(self, targetValue: tuple):
        if targetValue is None:
            self._imageWidget.setTargetPx((-1, -1))
            self._positionLabel.setText("")
        else:
            self._imageWidget.setTargetReal(targetValue)
            self.updatePositionLabel(targetValue, (False, False))
