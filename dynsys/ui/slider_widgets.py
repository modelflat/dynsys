from PyQt5 import Qt, QtCore
from PyQt5.Qt import QSlider, pyqtSignal as Signal


class RealSlider(QSlider):
    valueChanged = Signal(float)

    def __init__(self, min_val, max_val, steps=10000, horizontal=True):
        super().__init__()
        self.steps = steps
        self.min_val = min_val
        self.max_val = max_val
        self.setOrientation(QtCore.Qt.Vertical if not horizontal else QtCore.Qt.Horizontal)
        self.setMinimum(0)
        self.setMaximum(self.steps)
        self.valueChanged.connect(self._value_changed)

    @QtCore.pyqtSlot(float, name="_value_changed")
    def _value_changed(self, _):
        self.valueChanged.emit(self.value())

    def set_value(self, v):
        super().setValue(int((v - self.min_val) / (self.max_val - self.min_val) * self.steps))

    def value(self):
        return float(super().value()) / self.steps * (self.max_val - self.min_val) + self.min_val

    @staticmethod
    def makeAndConnect(min_val, max_val, current_val=None, steps=10000,
                       horizontal=True, connect_to=None):
        s = RealSlider(min_val, max_val, steps=steps, horizontal=horizontal)
        if connect_to is not None:
            s.valueChanged.connect(connect_to)
        s.set_value(current_val if current_val is not None else min_val)
        return s


class IntegerSlider(QSlider):

    def __init__(self, min_val, max_val, horizontal=True):
        super().__init__()
        self.setOrientation(QtCore.Qt.Vertical if not horizontal else QtCore.Qt.Horizontal)
        self.setMinimum(min_val)
        self.setMaximum(max_val)

    @staticmethod
    def makeAndConnect(min_val, max_val, current_val=None, horizontal=True, connect_to=None):
        s = IntegerSlider(min_val, max_val, horizontal=horizontal)
        if connect_to is not None:
            s.valueChanged.connect(connect_to)
        s.setValue(current_val if current_val is not None else min_val)
        return s
