import sys

from typing import Union

from PyQt5.Qt import QWidget, QApplication
from PyQt5.Qt import QObject, pyqtSignal as Signal

from .cl import *
from .ui import *
from .PhasePlot import PhasePlot
from .ParameterSurface import ParameterSurface
from .CobwebDiagram import CobwebDiagram
from .ParameterMap import ParameterMap
from .BifurcationTree import BifurcationTree
from .BasinsOfAttraction import BasinsOfAttraction


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
        import json
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
                    self.config = json.load(f)
            except Exception as e:
                raise RuntimeWarning("Cannot load configuration from file %s: %s" % (self.configFile, str(e)))
            else:
                print("Loaded configuration:", self.config)

        self.ctx, self.queue = createContextAndQueue(self.config)

    def _convertBounds(self, bounds):
        return bounds.asTuple() if type(bounds) is Bounds else bounds

    def makePhasePlot(self, source: str, paramCount: int,
                      spaceShape: Union[tuple, Bounds] = (-1., 1., -1., 1.),
                      imageShape: tuple = (512, 512),
                      backColor: tuple = (1.0, 1.0, 1.0, 1.0),
                      typeConfig: TypeConfig = FLOAT,
                      withUi: bool = True,
                      uiNames: tuple = ("x", "y"),
                      uiShape: tuple = (False, False),
                      uiTargetColor: QColor = Qt.red
                      ):
        bounds = self._convertBounds(spaceShape)
        plot = PhasePlot(
            self.ctx, self.queue, imageShape, bounds, source,
            paramCount=paramCount, backColor=backColor, typeConfig=typeConfig
        )
        ui = ParameterizedImageWidget(bounds, uiNames, uiShape, uiTargetColor) if withUi else None
        return plot if ui is None else (plot, ui)

    def makeParameterSurface(self, source: str,
                             spaceShape: Union[tuple, Bounds] = (-1., 1., -1., 1.),
                             imageShape: tuple = (512, 512),
                             typeConfig: TypeConfig = FLOAT,
                             withUi: bool = True,
                             uiNames: tuple = None,
                             uiShape: tuple = None,
                             uiTargetColor: QColor = Qt.red
                             ):
        bounds = self._convertBounds(spaceShape)
        surf = ParameterSurface(self.ctx, self.queue, imageShape, bounds, source, typeConfig=typeConfig)
        ui = ParameterizedImageWidget(bounds, uiNames, uiShape, uiTargetColor) if withUi else None
        return surf if ui is None else (surf, ui)

    def makeCobwebDiagram(self, source: str, paramCount: int,
                          spaceShape: Union[tuple, Bounds] = (-1., 1., -1., 1.),
                          imageShape: tuple = (512, 512),
                          typeConfig: TypeConfig = FLOAT,
                          withUi: bool = True,
                          uiNames: tuple = None,
                          uiShape: tuple = None,
                          uiTargetColor: QColor = Qt.red
                          ):
        bounds = self._convertBounds(spaceShape)
        diag = CobwebDiagram(
            self.ctx, self.queue, imageShape, bounds, source,
            paramCount=paramCount, typeConfig=typeConfig
        )
        ui = ParameterizedImageWidget(bounds, uiNames, uiShape, uiTargetColor) if withUi else None
        return diag if ui is None else (diag, ui)

    def makeParameterMap(self, source: str, variableCount: int,
                         spaceShape: Union[tuple, Bounds] = (-1., 1., -1., 1.),
                         imageShape: tuple = (512, 512),
                         typeConfig: TypeConfig = FLOAT,
                         withUi: bool = True,
                         uiNames: tuple = None,
                         uiShape: tuple = None,
                         uiTargetColor: QColor = Qt.red
                         ):
        bounds = self._convertBounds(spaceShape)
        pmap = ParameterMap(
            self.ctx, self.queue, imageShape, bounds, source,
            varCount=variableCount, typeConfig=typeConfig
        )
        ui = ParameterizedImageWidget(bounds, uiNames, uiShape, uiTargetColor) if withUi else None
        return pmap if ui is None else (pmap, ui)

    def makeBifurcationTree(self, source, paramCount: int,
                            paramRange: Union[tuple, Bounds] = (-1., 1.),
                            imageShape: tuple = (512, 512),
                            typeConfig: TypeConfig = FLOAT,
                            withUi: bool = True,
                            uiNames: tuple = None,
                            uiShape: tuple = None,
                            uiTargetColor: QColor = Qt.red
                            ):
        bounds = self._convertBounds(paramRange)
        bif = BifurcationTree(
            self.ctx, self.queue, imageShape, source,
            paramCount=paramCount, typeConfig=typeConfig
        )
        ui = ParameterizedImageWidget(bounds, uiNames, uiShape, uiTargetColor) if withUi else None
        return bif if ui is None else (bif, ui)

    def makeBasinsOfAttraction(self, source, paramCount: int,
                               spaceShape: Union[tuple, Bounds] = (-1., 1., -1., 1.),
                               imageShape: tuple = (512, 512),
                               typeConfig: TypeConfig = FLOAT,
                               withUi: bool = True,
                               uiNames: tuple = None,
                               uiShape: tuple = None,
                               uiTargetColor: QColor = Qt.red
                               ):
        bounds = self._convertBounds(spaceShape)
        boa = BasinsOfAttraction(
            self.ctx, self.queue, imageShape, bounds, source,
            paramCount=paramCount, typeConfig=typeConfig
        )
        ui = ParameterizedImageWidget(bounds, uiNames, uiShape, uiTargetColor) if withUi else None
        return boa if ui is None else (boa, ui)

    def run(self):
        self.show()
        sys.exit(self.app.exec_())
