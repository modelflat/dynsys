from typing import Union, Iterable, Callable

from PyQt5.Qt import pyqtSignal as Signal
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLayout, QLabel, QSlider


class RealSlider(QSlider):
    valueChanged = Signal(float)

    def __init__(self, bounds: tuple, horizontal=True, steps=10000):
        super().__init__()
        self._steps = steps
        self._bounds = bounds
        self.setOrientation(Qt.Vertical if not horizontal else Qt.Horizontal)
        self.setMinimum(0)
        self.setMaximum(self._steps)

        super().valueChanged.connect(lambda _: self.valueChanged.emit(self.value()))

    def setValue(self, v):
        super().setValue(int((v - self._bounds[0]) / (self._bounds[1] - self._bounds[0]) * self._steps))

    def value(self):
        return float(super().value()) / self._steps * (self._bounds[1] - self._bounds[0]) + self._bounds[0]


class IntegerSlider(QSlider):

    def __init__(self, bounds: tuple, horizontal=True):
        super().__init__()
        self.setOrientation(Qt.Vertical if not horizontal else Qt.Horizontal)
        self.setMinimum(bounds[0])
        self.setMaximum(bounds[1])


def createSlider(sliderType: str, bounds: tuple,
                 horizontal: bool = True,
                 withLabel: str = None, labelPosition: str = "left",
                 withValue: Union[float, int, None] = None,
                 connectTo: Union[Iterable, Callable, None] = None,
                 putToLayout: QLayout = None
                 ) -> tuple:
    sliderType = sliderType.lower()
    if sliderType in {"real", "r", "float"}:
        slider = RealSlider(bounds, horizontal)
    elif sliderType in {"int", "integer", "i", "d"}:
        slider = IntegerSlider(bounds, horizontal)
    else:
        raise ValueError("Unknown slider type: {}".format(sliderType))
    if withValue is not None:
        slider.setValue(withValue)

    layout = None
    if withLabel is not None:
        positions = {"left", "top", "right"}
        if labelPosition not in positions:
            raise ValueError("Label position must be one of: {}".format(positions))
        supportsValues = True
        try:
            withLabel.format(0.42)
        except:
            supportsValues = False
        label = QLabel(withLabel)

        if supportsValues:
            def setVal(val): label.setText(withLabel.format(val))

            setVal(withValue)
            slider.valueChanged.connect(setVal)

        if putToLayout is None:
            layout = (QHBoxLayout if labelPosition in {"left", "right"} else QVBoxLayout)()
        else:
            layout = putToLayout
        if labelPosition in {"right"}:
            layout.addWidget(slider)
            layout.addWidget(label)
        else:
            layout.addWidget(label)
            layout.addWidget(slider)

    if connectTo is not None:
        if isinstance(connectTo, Iterable):
            [slider.valueChanged.connect(fn) for fn in connectTo]
        else:
            slider.valueChanged.connect(connectTo)

    return slider, slider if layout is None else layout
