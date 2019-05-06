from PySide2.QtCore import *
from PySide2.QtWidgets import *


def qt_layout(
        *objects,
        orientation: Qt.Orientation = Qt.Vertical,
        alignment: Qt.Alignment = None
) -> QVBoxLayout:
    if orientation == Qt.Horizontal:
        layout = QHBoxLayout()
    else:
        layout = QVBoxLayout()

    if alignment is not None:
        layout.setAlignment(alignment)

    for obj in objects:
        if isinstance(obj, QLayout):
            layout.addLayout(obj)
        if isinstance(obj, QWidget):
            layout.addWidget(obj)

    return layout


def qt_splitter(
        *objects,
        orientation: Qt.Orientation = Qt.Vertical,
        sizes: list = None
) -> QSplitter:
    splitter = QSplitter(orientation)
    for obj in objects:
        if isinstance(obj, QWidget):
            splitter.addWidget(obj)
        if isinstance(obj, QLayout):
            wdg = QWidget()
            wdg.setLayout(obj)
            splitter.addWidget(wdg)

        if sizes is not None:
            splitter.setSizes(sizes)

    return splitter


def qt_line(
        orientation: Qt.Orientation = Qt.Horizontal,
        shadow: QFrame.Shadow = QFrame.Plain
):
    line = QFrame()
    if orientation == Qt.Horizontal:
        line.setFrameShape(QFrame.HLine)
    elif orientation == Qt.Vertical:
        line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(shadow)
    return line