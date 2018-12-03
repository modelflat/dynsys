import abc
import numpy

from PyQt5 import QtCore
from PyQt5.Qt import QVector3D, QImage, QPixmap, QColor, QPainter, QPen, pyqtSignal as Signal
from PyQt5.QtDataVisualization import QCustom3DVolume, Q3DScatter, Q3DTheme, Q3DCamera, QAbstract3DGraph, Q3DInputHandler
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtGui import QMouseEvent


def toPixmap(data: numpy.ndarray):
    image = QImage(data.data, *data.shape[:-1], QImage.Format_ARGB32)
    pixmap = QPixmap()
    # noinspection PyArgumentList
    pixmap.convertFromImage(image)
    return pixmap


def mouseButtonsState(mouseEvent):
    return bool(QtCore.Qt.LeftButton & mouseEvent.buttons()), bool(QtCore.Qt.RightButton & mouseEvent.buttons())


class ImageWidget(QLabel):

    selectionChanged = Signal(tuple, tuple)

    @abc.abstractmethod
    def spaceShape(self) -> tuple: ...

    @abc.abstractmethod
    def setSpaceShape(self, spaceShape: tuple) -> None: ...

    @abc.abstractmethod
    def textureShape(self): ...

    @abc.abstractmethod
    def setTexture(self, data: numpy.ndarray) -> None: ...

    @abc.abstractmethod
    def targetPx(self) -> tuple: ...

    @abc.abstractmethod
    def setTargetPx(self, targetLocation: tuple) -> None: ...

    @abc.abstractmethod
    def targetReal(self) -> tuple: ...

    @abc.abstractmethod
    def setTargetReal(self, targetLocation: tuple) -> None: ...


class Target2D:

    def __init__(self, color: QColor, shape: tuple = (True, True)):
        self._color = color
        self._shape = shape
        self._pos = (-1, -1)
        self._setPosCalled = False

    def setPosCalled(self):
        return self._setPosCalled

    def shape(self):
        return self._shape

    def pos(self):
        return self._pos

    def setPos(self, pos: tuple):
        self._setPosCalled = True
        self._pos = pos

    def draw(self, w: int, h: int, painter: QPainter):
        pen = QPen(self._color, 1)
        painter.setPen(pen)
        if self._shape[1]:
            painter.drawLine(0, self._pos[1], w, self._pos[1])
        if self._shape[0]:
            painter.drawLine(self._pos[0], 0, self._pos[0], h)


class Image2D(ImageWidget):

    def _onMouseEvent(self, event):
        left, right = mouseButtonsState(event)
        if left:
            self._target.setPos((event.x(), event.y()))
            self.repaint()
            if self._target.setPosCalled():
                vals = self.targetReal()
                self.selectionChanged.emit(
                    (vals[0] if self._target.shape()[0] else None,
                     vals[1] if self._target.shape()[1] else None),
                    (left, right))

    def mousePressEvent(self, event):
        super(Image2D, self).mousePressEvent(event)
        self._onMouseEvent(event)

    def mouseMoveEvent(self, event):
        super(Image2D, self).mouseMoveEvent(event)
        self._onMouseEvent(event)

    def paintEvent(self, QPaintEvent):
        super(ImageWidget, self).paintEvent(QPaintEvent)
        self._target.draw(self.width(), self.height(), QPainter(self))

    def __init__(self, targetColor: QColor = QtCore.Qt.red,
                 targetShape: tuple = (True, True),
                 spaceShape: tuple = (-1.0, 1.0, -1.0, 1.0),
                 invertY: bool = True
                 ):
        super().__init__()
        self.setMouseTracking(True)
        self._target = Target2D(targetColor, targetShape)
        self._spaceShape = spaceShape
        self._invertY = invertY
        self._textureShape = (1, 1)
        self._textureDataReference = None

    def spaceShape(self) -> tuple:
        return self._spaceShape

    def setSpaceShape(self, spaceShape: tuple) -> None:
        self._spaceShape = spaceShape

    def textureShape(self) -> tuple:
        return self._textureShape

    def setTexture(self, data: numpy.ndarray) -> None:
        self._textureDataReference = data
        self._textureShape = data.shape[:-1]
        self.setPixmap(toPixmap(data))

    def targetPx(self) -> tuple:
        return self._target.pos()

    def setTargetPx(self, targetLocation: tuple) -> None:
        self._target.setPos(targetLocation)

    def targetReal(self) -> tuple:
        x, y = self._target.pos()
        x = self._spaceShape[0] + x / self._textureShape[0] * (self._spaceShape[1] - self._spaceShape[0])
        if self._invertY:
            y = self._spaceShape[2] + (self._textureShape[1] - y) /\
                self._textureShape[1] * (self._spaceShape[3] - self._spaceShape[2])
        else:
            y = self._spaceShape[2] + y / self._textureShape[1] * (self._spaceShape[3] - self._spaceShape[2])
        x = numpy.clip(x, self._spaceShape[0], self._spaceShape[1])
        y = numpy.clip(y, self._spaceShape[2], self._spaceShape[3])
        return x, y

    def setTargetReal(self, targetLocation: tuple) -> None:
        x, y = targetLocation
        x = (x - self._spaceShape[0]) / (self._spaceShape[1] - self._spaceShape[0])*self._textureShape[0]
        y = self._textureShape[1] - (y - self._spaceShape[2]) /\
            self._textureShape[1] * (self._spaceShape[3] - self._spaceShape[2])
        self._target.setPos((x, y))
        self.repaint()


def exchangeLeftAndRightButtonState(event: QMouseEvent) -> QMouseEvent:
    if event.button() == QtCore.Qt.LeftButton:
        event = QMouseEvent(event.type(), event.pos(), QtCore.Qt.RightButton, event.buttons(), event.modifiers())
    elif event.button() == QtCore.Qt.RightButton:
        event = QMouseEvent(event.type(), event.pos(), QtCore.Qt.LeftButton, event.buttons(), event.modifiers())
    return event


class Custom3DInputHander(Q3DInputHandler):

    def mousePressEvent(self, event, QPoint):
        super().mousePressEvent(exchangeLeftAndRightButtonState(event), QPoint)

    def mouseReleaseEvent(self, event, QPoint):
        super().mouseReleaseEvent(exchangeLeftAndRightButtonState(event), QPoint)

    def mouseMoveEvent(self, event: QMouseEvent, QPoint):
        super().mouseMoveEvent(exchangeLeftAndRightButtonState(event), QPoint)


class Image3D(ImageWidget):

    def __init__(self,
                 spaceShape: tuple = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
                 segmentShape: tuple = (8, 8, 8),
                 swapRightAndLeftButtons=True):
        super().__init__()

        self._graph = Q3DScatter()
        if swapRightAndLeftButtons:
            self._graph.setActiveInputHandler(Custom3DInputHander())
        self._graph.setOrthoProjection(True)
        self._graph.activeTheme().setType(Q3DTheme.ThemeQt)
        self._graph.activeTheme().setBackgroundEnabled(False)
        self._graph.setShadowQuality(QAbstract3DGraph.ShadowQualityNone)
        self._graph.activeInputHandler().setZoomAtTargetEnabled(False)
        self._graph.scene().activeCamera().setCameraPreset(Q3DCamera.CameraPresetIsometricLeft)
        self._graph.scene().activeCamera().setZoomLevel(180)
        self._graph.axisX().setSegmentCount(segmentShape[0])
        self._graph.axisX().setTitle("X")
        self._graph.axisX().setTitleVisible(True)
        self._graph.axisY().setSegmentCount(segmentShape[1])
        self._graph.axisY().setTitle("Z")
        self._graph.axisY().setTitleVisible(True)
        self._graph.axisZ().setSegmentCount(segmentShape[2])
        self._graph.axisZ().setTitle("Y")
        self._graph.axisZ().setTitleVisible(True)
        self._graph.setAspectRatio(1)

        self._volume = QCustom3DVolume()
        self._volume.setUseHighDefShader(True)
        self._volume.setAlphaMultiplier(1.0)
        self._volume.setPreserveOpacity(False)
        self._volume.setDrawSlices(False)
        self._volume.setScaling(QVector3D(1.0, 1.0, 1.0))

        self._spaceShape = None
        self.setSpaceShape(spaceShape)

        self._currentTexture = numpy.empty((0, 0, 0, 4), dtype=numpy.uint8)

        self._graph.addCustomItem(self._volume)
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(QWidget().createWindowContainer(self._graph))

    def spaceShape(self) -> tuple:
        return self._spaceShape

    def setSpaceShape(self, spaceShape):
        self._spaceShape = spaceShape
        boundingBox = min(spaceShape[::2]), max(spaceShape[1::2])
        self._graph.axisX().setRange(*boundingBox)
        self._graph.axisY().setRange(*boundingBox)
        self._graph.axisZ().setRange(*boundingBox)
        pos = (boundingBox[1] + boundingBox[0]) / 2.0
        self._volume.setPosition(QVector3D(
            pos, pos, pos
        ))

    def textureShape(self):
        return self._currentTexture.shape

    def setTexture(self, data: numpy.ndarray):
        if len(data.shape) != 4:
            raise RuntimeError("textureShape shoud have 4 components and be in form (w, h, d, 4)")
        self._currentTexture = data
        self._volume.setTextureFormat(QImage.Format_ARGB32)
        self._volume.setTextureDimensions(*data.shape[:-1])
        self._volume.setTextureData(data.tobytes())

    def targetPx(self) -> tuple:
        raise RuntimeError("Image3D supports sink mode only")

    def setTargetPx(self, targetLocation: tuple) -> None:
        raise NotImplementedError()

    def targetReal(self) -> tuple:
        raise RuntimeError("Image3D supports sink mode only")

    def setTargetReal(self, targetLocation: tuple) -> None:
        raise NotImplementedError()


if __name__ == '__main__':

    def testWidget(fn):
        from PyQt5.Qt import QApplication, QHBoxLayout
        app = QApplication([])
        win = QWidget()
        win.setFixedSize(512, 512)
        layout = QHBoxLayout()
        layout.addWidget(fn())
        win.setLayout(layout)
        win.show()
        exit(app.exec())

    def test2D():
        w = Image2D()
        w.selectionChanged.connect(lambda *args: print(*args))
        return w

    def test3D():
        w = Image3D()
        w.setTexture(numpy.random.randint(0, 0xFF, size=(2, 2, 2, 4), dtype=numpy.uint8))
        return w

    # testWidget(test2D)
    testWidget(test3D)
