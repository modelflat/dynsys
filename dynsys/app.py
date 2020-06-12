import argparse
import json
import logging
import sys
import abc

import numpy
from PySide2 import QtCore
from PySide2.QtGui import QVector3D, QImage, QPixmap, QColor, QPainter, QPen
from PySide2.QtDataVisualization import QtDataVisualization
from PySide2.QtGui import QMouseEvent
from PySide2.QtWidgets import QApplication, QDesktopWidget, QVBoxLayout, QLayout
from PySide2.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide2.QtCore import Signal


from core import create_context_and_queue


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocl-config", required=False, default=None, type=str,
                        help="Configuration for OpenCL context (JSON string)")
    parser.add_argument("--ocl-config-path", required=False, default=None, type=str,
                        help="Path to config file for OpenCL context")
    return parser.parse_args(sys.argv[1:])


class SimpleApp(QWidget):
    """
    Simple Qt application. Includes empty Qt window (centered on screen) and OpenCL context/command queue
    Arguments hinting on OpenCL configuration can be supplied to it, namely (in priority order):

    * ``--ocl-config`` - JSON string

    * ``--ocl-config-path`` - path to JSON file with config

    Sample usage:

    >>> from dynsys.app import SimpleApp
    >>> class MyApp(SimpleApp):
    ...     def __init__(self):
    ...         super(MyApp, self).__init__("My Application")
    >>> MyApp()#.run()

    (``.run()`` is commented out here because it will fail inside doctest.)

    This will spawn and show empty window titled "My Application". You can use the overridden constructor to setup
    your application, and Qt's signal/slot system to make it responsive.
    """

    ctx = property(lambda self: self._ctx)
    queue = property(lambda self: self._queue)

    def __init__(self, title):
        """
        :param title: Window title
        """
        self._app = QApplication(sys.argv)

        # noinspection PyArgumentList
        super().__init__()

        self.setWindowTitle(title)

        ns = _parse_args()

        if ns.ocl_config is not None:
            config_str = ns.ocl_config
        elif ns.ocl_config_path is not None:
            try:
                with open(ns.ocl_config_path) as f:
                    config_str = f.read()
            except IOError as e:
                logging.error("Error reading config file", e)
        else:
            config_str = None

        if config_str is not None:
            try:
                config = json.loads(config_str)
            except Exception as e:
                config = None
                logging.error("Error parsing config", e)
            else:
                logging.info(f"Loaded config: {config_str}")
        else:
            config = None
            logging.info("No config supplied, will try to autodetect CL platform/device")

        self._ctx, self._queue = create_context_and_queue(config)

    def run(self):
        """
        Run this application.
        """
        screen = QDesktopWidget().screenGeometry()
        self.show()
        x = ((screen.width() - self.width()) // 2) if screen.width() > self.width() else 0
        y = ((screen.height() - self.height()) // 2) if screen.height() > self.height() else 0
        self.move(x, y)
        sys.exit(self._app.exec_())


def stack(*widgets_or_layouts, kind="v"):
    """
    Stack widgets vertically or horizontally into QVBoxLayout or QHBoxLayout correspondingly

    :param widgets_or_layouts: widgets/layouts to stack (can be mixed)
    :param kind: how to stack ("v" or "h")
    :return: layout
    """
    if kind == "v":
        layout = QVBoxLayout()
    elif kind == "h":
        layout = QHBoxLayout()
    else:
        raise RuntimeError(f"No such stack kind: '{kind}'")

    for a in widgets_or_layouts:
        if isinstance(a, QLayout):
            layout.addLayout(a)
        else:
            layout.addWidget(a)
    return layout


def to_pixmap(data: numpy.ndarray):
    image = QImage(data.data, *data.shape[:-1], QImage.Format_ARGB32)
    pixmap = QPixmap()

    # noinspection PyArgumentList
    pixmap.convertFromImage(image)

    return pixmap


def mouse_buttons_state(event):
    return bool(QtCore.Qt.LeftButton & event.buttons()), bool(QtCore.Qt.RightButton & event.buttons())


class ImageWidget(QLabel):

    selection_changed = Signal(tuple, tuple)

    @abc.abstractmethod
    def space_shape(self) -> tuple: ...

    @abc.abstractmethod
    def set_space_shape(self, space_shape: tuple) -> None: ...

    @abc.abstractmethod
    def texture_shape(self): ...

    @abc.abstractmethod
    def set_texture(self, data: numpy.ndarray) -> None: ...

    @abc.abstractmethod
    def target_px(self) -> tuple: ...

    @abc.abstractmethod
    def set_target_px(self, target: tuple) -> None: ...

    @abc.abstractmethod
    def target(self) -> tuple: ...

    @abc.abstractmethod
    def set_target(self, target: tuple) -> None: ...


class Target2D:

    def __init__(self, color: QColor, shape: tuple = (True, True)):
        self.color = color
        self.shape = shape
        self.pos = (-1, -1)
        self.set_pos_called = False

    def set_pos(self, pos: tuple):
        self.set_pos_called = True
        self.pos = pos

    def draw(self, w: int, h: int, painter: QPainter):
        pen = QPen(self.color, 1)
        painter.setPen(pen)
        if self.shape[1]:
            painter.drawLine(0, self.pos[1], w, self.pos[1])
        if self.shape[0]:
            painter.drawLine(self.pos[0], 0, self.pos[0], h)


class Image2D(ImageWidget):

    def __init__(self,
                 target_color: QColor = QtCore.Qt.red,
                 target_shape: tuple = (True, True),
                 space_shape: tuple = (-1.0, 1.0, -1.0, 1.0),
                 texture_shape: tuple = (1, 1),
                 invert_y: bool = True
                 ):
        super().__init__()
        self.setMouseTracking(True)
        self._target = Target2D(target_color, target_shape)
        self._spaceShape = space_shape
        self._invertY = invert_y
        self._textureShape = texture_shape
        self._textureDataReference = None

    def _on_mouse_event(self, event):
        if not any(self._target.shape):
            return
        left, right = mouse_buttons_state(event)
        if left:
            self._target.set_pos((event.x(), event.y()))
            self.repaint()
            if self._target.set_pos_called:
                vals = self.target()
                self.selection_changed.emit(
                    (vals[0] if self._target.shape[0] else None,
                     vals[1] if self._target.shape[1] else None),
                    (left, right))

    def mousePressEvent(self, event):
        super(Image2D, self).mousePressEvent(event)
        self._on_mouse_event(event)

    def mouseMoveEvent(self, event):
        super(Image2D, self).mouseMoveEvent(event)
        self._on_mouse_event(event)

    def paintEvent(self, QPaintEvent):
        super(ImageWidget, self).paintEvent(QPaintEvent)
        self._target.draw(self.width(), self.height(), QPainter(self))

    def space_shape(self) -> tuple:
        return self._spaceShape

    def set_space_shape(self, space_shape: tuple) -> None:
        self._spaceShape = space_shape

    def texture_shape(self) -> tuple:
        return self._textureShape

    def set_texture(self, data: numpy.ndarray) -> None:
        self._textureDataReference = data
        self._textureShape = data.shape[:-1]
        self.setPixmap(to_pixmap(data))

    def target_px(self) -> tuple:
        return self._target.pos

    def set_target_px(self, target_location: tuple) -> None:
        self._target.set_pos(target_location)

    def target(self) -> tuple:
        x, y = self._target.pos
        x = self._spaceShape[0] + x / self._textureShape[0] * (self._spaceShape[1] - self._spaceShape[0])
        if self._invertY:
            y = self._spaceShape[2] + (self._textureShape[1] - y) / \
                self._textureShape[1] * (self._spaceShape[3] - self._spaceShape[2])
        else:
            y = self._spaceShape[2] + y / self._textureShape[1] * (self._spaceShape[3] - self._spaceShape[2])
        x = numpy.clip(x, self._spaceShape[0], self._spaceShape[1])
        y = numpy.clip(y, self._spaceShape[2], self._spaceShape[3])
        return x, y

    def set_target(self, target_location: tuple) -> None:
        x, y = target_location
        x = (x - self._spaceShape[0]) / (self._spaceShape[1] - self._spaceShape[0])*self._textureShape[0]
        y = self._textureShape[1] - \
            (y - self._spaceShape[2]) / (self._spaceShape[3] - self._spaceShape[2])*self._textureShape[1]
        self._target.set_pos((x, y))
        self.repaint()


def exchange_left_and_right_button_state(event: QMouseEvent) -> QMouseEvent:
    if event.button() == QtCore.Qt.LeftButton:
        event = QMouseEvent(event.type(), event.pos(), QtCore.Qt.RightButton, event.buttons(), event.modifiers())
    elif event.button() == QtCore.Qt.RightButton:
        event = QMouseEvent(event.type(), event.pos(), QtCore.Qt.LeftButton, event.buttons(), event.modifiers())
    return event


class Custom3DInputHander(QtDataVisualization.Q3DInputHandler):

    def mousePressEvent(self, event, QPoint):
        super().mousePressEvent(exchange_left_and_right_button_state(event), QPoint)

    def mouseReleaseEvent(self, event, QPoint):
        super().mouseReleaseEvent(exchange_left_and_right_button_state(event), QPoint)

    def mouseMoveEvent(self, event: QMouseEvent, QPoint):
        super().mouseMoveEvent(exchange_left_and_right_button_state(event), QPoint)


class Image3D(ImageWidget):

    def __init__(self,
                 space_shape: tuple = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
                 segment_shape: tuple = (8, 8, 8),
                 swap_right_and_left_buttons=True,
                 use_theme=QtDataVisualization.Q3DTheme.ThemeQt
                 ):
        super().__init__()

        self._graph = QtDataVisualization.Q3DScatter()
        if swap_right_and_left_buttons:
            self._graph.setActiveInputHandler(Custom3DInputHander())
        self._graph.setOrthoProjection(True)
        self._graph.activeTheme().setType(use_theme)
        self._graph.activeTheme().setBackgroundEnabled(True)
        self._graph.setShadowQuality(QtDataVisualization.QAbstract3DGraph.ShadowQualityNone)
        self._graph.activeInputHandler().setZoomAtTargetEnabled(False)
        self._graph.scene().activeCamera().setCameraPreset(QtDataVisualization.Q3DCamera.CameraPresetIsometricLeft)
        self._graph.scene().activeCamera().setZoomLevel(180)
        self._graph.axisX().setSegmentCount(1)#segment_shape[0])
        self._graph.axisX().setTitle("X")
        self._graph.axisX().setTitleVisible(True)
        self._graph.axisX().setAutoAdjustRange(True)

        self._graph.axisY().setSegmentCount(1)#segment_shape[1])
        self._graph.axisY().setTitle("Z")
        self._graph.axisY().setTitleVisible(True)
        self._graph.axisY().setAutoAdjustRange(True)

        self._graph.axisZ().setSegmentCount(1)#segment_shape[2])
        self._graph.axisZ().setTitle("Y")
        self._graph.axisZ().setTitleVisible(True)
        self._graph.axisZ().setAutoAdjustRange(True)

        # self._graph.setAspectRatio(1)

        self._volume = QtDataVisualization.QCustom3DVolume()
        self._volume.setUseHighDefShader(True)
        self._volume.setAlphaMultiplier(1.0)
        self._volume.setPreserveOpacity(False)
        self._volume.setDrawSlices(False)
        self._volume.setScaling(QVector3D(1.0, 1.0, 1.0))

        self._spaceShape = None
        self.set_space_shape(space_shape)

        self._current_texture = numpy.empty((0, 0, 0, 4), dtype=numpy.uint8)

        self._graph.addCustomItem(self._volume)
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(QWidget().createWindowContainer(self._graph))

    def space_shape(self) -> tuple:
        return self._spaceShape

    def set_space_shape(self, space_shape):
        self._spaceShape = space_shape
        bounding_box = min(space_shape[::2]), max(space_shape[1::2])
        self._graph.axisX().setRange(*space_shape[0:2])
        self._graph.axisY().setRange(*space_shape[2:4])
        self._graph.axisZ().setRange(*space_shape[4:6])
        # pos = (bounding_box[1] + bounding_box[0]) / 2.0
        self._volume.setPosition(QVector3D(
            (space_shape[0] + space_shape[1]) / 2.0,
            (space_shape[2] + space_shape[3]) / 2.0,
            (space_shape[4] + space_shape[5]) / 2.0,
            ))

    def texture_shape(self):
        return self._current_texture.shape

    def set_texture(self, data: numpy.ndarray):
        if len(data.shape) != 4:
            raise RuntimeError("textureShape shoud have 4 components and be in form (w, h, d, 4)")
        self._current_texture = data
        self._volume.setTextureFormat(QImage.Format_ARGB32)
        self._volume.setTextureDimensions(*data.shape[:-1])
        self._volume.setTextureData(data.tobytes())

    def target_px(self) -> tuple:
        raise RuntimeError("Image3D supports sink mode only")

    def set_target_px(self, target: tuple) -> None:
        raise NotImplementedError()

    def target(self) -> tuple:
        raise RuntimeError("Image3D supports sink mode only")

    def set_target(self, target: tuple) -> None:
        raise NotImplementedError()
