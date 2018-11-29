from .image_widgets import Image2D, Image3D
from .slider_widgets import RealSlider, IntegerSlider

from PyQt5.Qt import QVBoxLayout, QHBoxLayout, QLayout


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
