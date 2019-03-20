from multiprocessing import Lock

import numpy
import time
from PyQt5.QtWidgets import QLabel, QLineEdit, QCheckBox, QPushButton, QComboBox

import config as cfg
from dynsys import SimpleApp, vStack, createSlider, hStack
from ifs_fractal import make_parameter_map, make_phase_plot, make_simple_param_surface, make_basins


class CourseWork(SimpleApp):

    def __init__(self):
        super().__init__("Coursework")
        self.compute_lock = Lock()

        self.phase, _ = \
            make_phase_plot(self.ctx, self.queue, cfg.phase_image_shape, cfg.phase_shape)

        self.basins, self.right_wgt = \
            make_basins(self.ctx, self.queue, cfg.basins_image_shape, cfg.phase_shape)

        self.param, self.param_wgt = \
            make_parameter_map(self.ctx, self.queue, cfg.param_map_image_shape, cfg.h_bounds, cfg.alpha_bounds)

        self.param_placeholder = \
            make_simple_param_surface(self.ctx, self.queue, cfg.param_map_image_shape, cfg.h_bounds, cfg.alpha_bounds)

        self.h_slider, self.h_slider_wgt = \
            createSlider("real", cfg.h_bounds, withLabel="h = {:2.3f}", labelPosition="top", withValue=0.5)

        self.alpha_slider, self.alpha_slider_wgt = \
            createSlider("real", cfg.alpha_bounds, withLabel="alpha = {:2.3f}", labelPosition="top", withValue=0.0)

        self.random_seq_gen_btn = QPushButton("Generate random (input length)")
        self.refresh_btn = QPushButton("Refresh")
        self.random_seq_reset_btn = QPushButton("Reset")
        self.param_map_compute_btn = QPushButton("Compute parameter map")
        self.param_map_draw_btn = QPushButton("Draw parameter map")
        self.clear_cb = QCheckBox("Clear image")
        self.clear_cb.setChecked(True)
        self.right_mode_cmb = QComboBox()

        self.period_label = QLabel()
        self.period_map = None

        self.root_seq_edit = QLineEdit()

        self.random_seq = None
        self.do_draw_basins = cfg.do_draw_basins

        self.right_wgts = {
            "phase":  self.compute_and_draw_phase,
            "basins": self.compute_and_draw_basins
        }
        self.right_mode_cmb.addItems(self.right_wgts.keys())

        self.setup_layout()
        self.connect_everything()
        self.draw_param_placeholder()
        self.draw_right()

    def setup_layout(self):
        left = vStack(
            self.param_map_compute_btn,
            hStack(self.param_wgt),
            hStack(self.random_seq_gen_btn, self.random_seq_reset_btn),
            self.root_seq_edit
        )
        left.setSpacing(1)
        right = vStack(
            hStack(self.right_mode_cmb, self.refresh_btn, self.clear_cb),
            self.right_wgt,
            self.period_label,
            self.alpha_slider_wgt,
            self.h_slider_wgt,
        )
        right.setSpacing(1)
        self.setLayout(hStack(left, right))
        self.layout().setSpacing(1)

    def connect_everything(self):

        def set_sliders_and_draw(val):
            self.draw_right()
            self.set_values_no_signal(*val)

        self.param_wgt.valueChanged.connect(set_sliders_and_draw)

        def set_h_value(h):
            _, alpha = self.param_wgt.value()
            self.param_wgt.setValue((h, alpha))
        self.h_slider.valueChanged.connect(set_h_value)

        def set_alpha_value(alpha):
            h, _ = self.param_wgt.value()
            self.param_wgt.valueChanged.emit((h, alpha))
        self.alpha_slider.valueChanged.connect(set_alpha_value)

        if cfg.param_map_draw_on_select and cfg.param_map_select_z0_from_phase:
            def select_z0(*_):
                self.compute_and_draw_param_map()
            self.right_wgt.valueChanged.connect(select_z0)

        def gen_random_seq_fn(*_):
            try:
                rootSeqSize = int(self.root_seq_edit.text())
                self.random_seq = numpy.random.randint(0, 2 + 1, size=rootSeqSize, dtype=numpy.int32)
                print(self.random_seq)
                self.root_seq_edit.setText(" ".join(map(str, self.random_seq)))
            except Exception as e:
                print(e)

        self.random_seq_gen_btn.clicked.connect(gen_random_seq_fn)

        def reset_random_seq_fn(*_):
            self.random_seq = None
        self.random_seq_reset_btn.clicked.connect(reset_random_seq_fn)

        self.right_mode_cmb.currentIndexChanged.connect(self.draw_right)
        self.refresh_btn.clicked.connect(self.draw_right)
        self.param_map_compute_btn.clicked.connect(self.compute_and_draw_param_map)

    def parse_root_sequence(self):
        try:
            raw = self.root_seq_edit.text()
            if self.random_seq is not None:
                return self.random_seq
            else:
                l = list(map(int, raw.split()))
                if len(l) == 0 or not all(map(lambda x: x <= 2, l)):
                    return None
                else:
                    return l
        except:
            return None

    def draw_param_placeholder(self):
        with self.compute_lock:
            self.param_wgt.setImage(self.param_placeholder())

    def compute_and_draw_basins(self, *_):
        h, alpha = self.param_wgt.value()

        # TODO why? probably some bug in parameter computation
        # alpha = cfg.alpha_bounds[1] - alpha

        print("Drawing basins for h = {:.6f}, alpha = {:.6f}".format(h, alpha))

        with self.compute_lock:
            self.basins.compute_points(
                alpha=alpha,
                h=h,
                c=cfg.C,
                skip=cfg.basins_skip,
                root_seq=self.parse_root_sequence(),
                resolution=cfg.basins_resolution
            )
            image = self.basins.draw_points(resolution=cfg.basins_resolution)
            self.right_wgt.setImage(image)

    def compute_and_draw_param_map(self, *_):
        with self.compute_lock:
            print("Start computing parameter map")
            t = time.perf_counter()

            if cfg.param_map_select_z0_from_phase:
                wgt = self.right_wgt if self.do_draw_basins else self.phase_wgt
                z0 = complex(*wgt.value())
            else:
                z0 = cfg.param_map_z0

            self.param.compute_points(
                z0=z0,
                c=cfg.C,
                skip=cfg.param_map_skip,
                iter=cfg.param_map_iter,
                tol=cfg.param_map_tolerance,
                root_seq=self.parse_root_sequence(),
                wait=True,
                resolution=cfg.param_map_resolution
            )
            t = time.perf_counter() - t
        print("Computed parameter map in {:.3f} s".format(t))

        with self.compute_lock:
            image, periods = self.param.display(num_points=cfg.param_map_iter,
                                                resolution=cfg.param_map_resolution)
            self.param_wgt.setImage(image)
            self.period_map = periods

    def compute_and_draw_phase(self, *_):
        h, alpha = self.param_wgt.value()

        with self.compute_lock:
            image = self.phase(
                alpha=alpha,
                h=h,
                c=cfg.C,
                grid_size=cfg.phase_grid_size,
                iterCount=cfg.phase_iter,
                skip=cfg.phase_skip,
                root_seq=self.parse_root_sequence(),
                clear_image=self.clear_cb.isChecked(),
                z0=cfg.phase_z0
            )
            self.right_wgt.setImage(image)

    def set_period_label(self):
        x_px, y_px = self.param_wgt._imageWidget.targetPx()
        x_px //= cfg.param_map_resolution
        y_px //= cfg.param_map_resolution
        if self.period_map is not None:
            x_px = max(min(cfg.param_map_image_shape[0] // cfg.param_map_resolution - 1, x_px), 0)
            y_px = max(min(cfg.param_map_image_shape[1] // cfg.param_map_resolution - 1, y_px), 0)
            print(x_px, y_px)
            y, x = int(y_px), int(x_px)
            per = self.period_map[y][x]
            self.period_label.setText(
                "Detected period: {}".format("<chaos({})>".format(per) if per > 0.25 * cfg.param_map_iter else per)
            )

    def set_values_no_signal(self, h, alpha):
        self.param_wgt.blockSignals(True)
        self.param_wgt.setValue((h, alpha))
        self.param_wgt.blockSignals(False)
        self.h_slider.blockSignals(True)
        self.h_slider.setValue(h)
        self.h_slider.blockSignals(False)
        self.alpha_slider.blockSignals(True)
        self.alpha_slider.setValue(alpha)
        self.alpha_slider.blockSignals(False)

    def draw_right(self):
        print("draw_right called")
        what = self.right_mode_cmb.currentText()
        self.set_period_label()
        self.right_wgts[what]()


if __name__ == '__main__':
    CourseWork().run()
