import os
import numpy
import sys

from PyQt5.Qt import QPalette
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QLineEdit, QCheckBox, QPushButton

from coursework import make_parameter_map, make_phase_plot, make_simple_param_surface
from dynsys import SimpleApp, ParameterSurface, vStack, createSlider, hStack

space_shape = (-1., 1., -1., 1.)
h_bounds = (-2, 2)
alpha_bounds = (0, 1)
phase_image_shape = (512, 512)
param_map_image_shape = (512, 512)


class CourseWork(SimpleApp):

    def __init__(self):
        super().__init__("Coursework")

        self.phase, self.phase_wgt = \
            make_phase_plot(self.ctx, self.queue, phase_image_shape, space_shape)

        self.param, self.param_wgt = \
            make_parameter_map(self.ctx, self.queue, param_map_image_shape, h_bounds, alpha_bounds)

        self.param_placeholder = \
            make_simple_param_surface(self.ctx, self.queue, param_map_image_shape, h_bounds, alpha_bounds)

        self.h_slider, self.h_slider_wgt = \
            createSlider("real", h_bounds, withLabel="h = {:2.3f}", labelPosition="top", withValue=0.5)

        self.alpha_slider, self.alpha_slider_wgt = \
            createSlider("real", alpha_bounds, withLabel="alpha = {:2.3f}", labelPosition="top", withValue=0.0)

        self.random_seq_gen_btn = QPushButton("Generate random (input length)")
        self.refresh_btn = QPushButton("Refresh")
        self.random_seq_reset_btn = QPushButton("Reset")
        self.clear_cb = QCheckBox("Clear image")
        self.clear_cb.setChecked(True)

        self.root_seq_edit = QLineEdit()

        self.random_seq = None

        self.setup_layout()
        self.connect_everything()

        # self.recompute_param_map()
        self.draw_param_placeholder()
        self.draw_phase()

    def connect_everything(self):
        def set_sliders_and_draw(val):
            self.draw_phase()
            self.h_slider.setValue(val[0])
            self.alpha_slider.setValue(val[1])
        self.param_wgt.valueChanged.connect(set_sliders_and_draw)

        def set_h_value(h):
            _, alpha = self.param_wgt.value()
            self.param_wgt.setValue((h, alpha))
            self.draw_phase()
        self.h_slider.valueChanged.connect(set_h_value)

        def set_alpha_value(alpha):
            h, _ = self.param_wgt.value()
            self.param_wgt.setValue((h, alpha))
            self.draw_phase()
        self.alpha_slider.valueChanged.connect(set_alpha_value)

        def gen_random_seq_fn(*_):
            try:
                rootSeqSize = int(self.root_seq_edit.text())
                self.random_seq = numpy.random.randint(0, 2 + 1, size=rootSeqSize, dtype=numpy.int32)
            except:
                pass
        self.random_seq_gen_btn.clicked.connect(gen_random_seq_fn)

        def reset_random_seq_fn(*_):
            self.random_seq = None
            self.dLabel.setText("[-1]")
        self.random_seq_reset_btn.clicked.connect(reset_random_seq_fn)

        self.refresh_btn.clicked.connect(self.draw_phase)

    def setup_layout(self):
        self.setLayout(
            hStack(
                vStack(
                    self.param_wgt,
                    hStack(self.random_seq_gen_btn, self.random_seq_reset_btn),
                    self.root_seq_edit
                ),
                vStack(
                    self.phase_wgt,
                    hStack(self.refresh_btn, self.clear_cb),
                    self.alpha_slider_wgt,
                    self.h_slider_wgt,
                )
            )
        )

    def parse_root_sequence(self):
        raw = self.root_seq_edit.text()
        if self.random_seq is not None:
            return self.random_seq
        else:
            l = list(map(int, raw.split()))
            if len(l) == 0 or not all(map(lambda x: x <= 2, l)):
                return None
            else:
                return l

    def draw_param_placeholder(self):
        self.param_wgt.setImage(self.param_placeholder())

    def recompute_param_map(self, *_):

        n_iter = 128
        n_skip = 256

        import time
        print("Start computing parameter map")
        t = time.perf_counter()
        self.param_map.compute_points(
            z0=complex(0.05, 0.05),
            c=complex(0.0, 0.5),
            skip=n_skip,
            iter=n_iter,
            tol=0.05,
            root_seq=None,
            wait=True
        )
        t = time.perf_counter() - t
        print("Computed parameter map in {:.3f} s".format(t))
        print("Trying to draw")
        t = time.perf_counter()
        image = self.param_map.display(
            num_points=n_iter
        )
        t = time.perf_counter() - t
        print("Drawn in {:.3f} s".format(t))

        self.alphaHParamSurfUi.setImage(image)

    def draw_phase(self, *_):
        h, alpha = self.param_wgt.value()

        try:
            seq = self.parse_root_sequence()
        except:
            seq = None

        image = self.phase(
            alpha=alpha,
            h=h,
            c=complex(-0.0, 0.5),
            grid_size=2,
            iterCount=8192 << 2,
            skip=0,
            root_seq=seq,
            clear_image=self.clear_cb.isChecked()
        )
        self.phase_wgt.setImage(image)


if __name__ == '__main__':
    CourseWork().run()
