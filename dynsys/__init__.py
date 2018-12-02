import sys
from json import load

from .cl.core import *
from .cl.codegen import *
from .ui import *

from .phase_plot import PhasePlot
from .parameter_surface import ParameterSurface
from .cobweb_diagram import CobwebDiagram
from .parameter_map import ParameterMap
from .bifurcation_tree import BifurcationTree
from .basins_of_attraction import BasinsOfAttraction

from PyQt5.Qt import QWidget, QApplication

import numpy as np
import pyopencl as cl

from PyQt5.Qt import QObject, pyqtSignal as Signal


class Bounds:

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def clamp_x(self, v):
        return np.clip(v, self.x_min, self.x_max)

    def from_integer_x(self, v, v_max):
        return self.x_min + v / v_max * (self.x_max - self.x_min)

    def clamp_y(self, v):
        return np.clip(v, self.y_min, self.y_max)

    def from_integer_y(self, v, v_max, invert=True):
        return self.y_min + ((v_max - v) if invert else v)/ v_max * (self.y_max - self.y_min)

    def to_integer(self, x, y, w, h, invert_y=True):
        y_val = int((y - self.y_min) / (self.y_max - self.y_min) * h)
        return (
            int((x - self.x_min) / (self.x_max - self.x_min) * w),
            y_val if not invert_y else h - y_val
        )

    def asTuple(self):
        return self.x_min, self.x_max, self.y_min, self.y_max

    @staticmethod
    def x(x_min, x_max):
        return Bounds(x_min, x_max, None, None)

    @staticmethod
    def y(y_min, y_max):
        return Bounds(None, None, y_min, y_max)


class ObservableValue(QObject):

    valueChanged = Signal(object)

    def __init__(self, initial):
        super().__init__()
        self._value = initial

    def value(self):
        return self._value

    def setValue(self, v):
        self._value = v
        self.valueChanged.emit(v)

    @staticmethod
    def makeAndConnect(initial, connect_to=None):
        o = ObservableValue(initial)
        if connect_to is not None:
            o.valueChanged.connect(connect_to)
        return o


class SimpleApp(QWidget):

    def __init__(self, title):
        self.app = QApplication(sys.argv)
        super().__init__(parent=None)
        self.setWindowTitle(title)
        self.configFile = None
        self.config = None
        if len(sys.argv) > 1:
            self.configFile = sys.argv[1]
            print("Loading config from file: ", self.configFile)
            try:
                with open(self.configFile) as f:
                    self.config = load(f)
            except Exception as e:
                raise RuntimeWarning("Cannot load configuration from file %s: %s" % (self.configFile, str(e)))
            else:
                print("Loaded configuration:", self.config)

        self.ctx, self.queue = createContextAndQueue(self.config)

    def makePhasePortrait(self, imageShape, spaceShape, systemFunction, paramCount, typeConfig=FLOAT):
        bounds = spaceShape if type(spaceShape) is tuple else spaceShape.asTuple()
        return PhasePlot(self.ctx, self.queue, imageShape, bounds, systemFunction,
                         paramCount=paramCount, typeConfig=typeConfig)

    def makeParameterSurface(self, bounds, colorFunctionSource, width=512, height=512, typeConfig=FLOAT):
        return ParameterSurface(self.ctx, self.queue, width, height, bounds, colorFunctionSource, typeConfig=typeConfig)

    def makeCobwebDiagram(self, bounds, carrying_function_source, param_count=1, width=512, height=512, type_config=FLOAT):
        return CobwebDiagram(self.ctx, self.queue, (width, height), bounds.asTuple(), carrying_function_source,
                             paramCount=param_count, typeConfig=type_config)

    def makeParameterMap(self, bounds, map_function_source, var_count=1, width=512, height=512, type_config=FLOAT):
        return ParameterMap(
            self.ctx, self.queue, (width, height), bounds.asTuple(),
            map_function_source,
            varCount=var_count,
            typeConfig=type_config
        )

    def makeBifurcationTree(self, map_function_source, param_count=1, width=512, height=512, type_config=FLOAT):
        return BifurcationTree(self.ctx, self.queue, (width, height), map_function_source,
                               paramCount=param_count, typeConfig=type_config)

    def makeBasinsOfAttraction(self, bounds, system_function_source, width=512, height=512, param_count=2, type_config=FLOAT):
        return BasinsOfAttraction(self.ctx, self.queue, width, height, bounds, system_function_source,
                                  param_count=param_count, type_config=type_config)

    def run(self):
        self.show()
        sys.exit(self.app.exec_())
