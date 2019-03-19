from multiprocessing import Lock

import numpy
import time
from PyQt5.QtWidgets import QLabel, QLineEdit, QCheckBox, QPushButton

import config as cfg
from dynsys import SimpleApp, vStack, createSlider, hStack
from ifs_fractal import make_parameter_map, make_phase_plot, make_simple_param_surface, make_basins


class CourseWork(SimpleApp):

    def __init__(self):
        super().__init__("Coursework")
        self.compute_lock = Lock()

        self.phase, self.phase_wgt = \
            make_phase_plot(self.ctx, self.queue, cfg.phase_image_shape, cfg.phase_shape)

        self.basins, self.basins_wgt = \
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

        self.period_label = QLabel()
        self.period_map = None

        self.root_seq_edit = QLineEdit()

        self.random_seq = None

        self.setup_layout()
        self.connect_everything()

        self.do_draw_basins = True

        self.draw_param_placeholder()
        if self.do_draw_basins:
            self.compute_and_draw_basins()
        else:
            self.draw_phase()

    def connect_everything(self):

        def draw():
            # print("Draw call!!!")
            if self.do_draw_basins:
                self.compute_and_draw_basins()
            else:
                self.draw_phase()

        def set_sliders_and_draw(val):
            draw()
            self.h_slider.setValue(val[0])
            self.alpha_slider.setValue(val[1])
        self.param_wgt.valueChanged.connect(set_sliders_and_draw)

        def set_h_value(h):
            _, alpha = self.param_wgt.value()
            self.param_wgt.setValue((h, alpha))
            # self.param_wgt.valueChanged.emit((h, alpha))
        self.h_slider.valueChanged.connect(set_h_value)

        def set_alpha_value(alpha):
            h, _ = self.param_wgt.value()
            self.param_wgt.setValue((h, alpha))
            # draw()
        self.alpha_slider.valueChanged.connect(set_alpha_value)

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

        self.refresh_btn.clicked.connect(draw)
        self.param_map_compute_btn.clicked.connect(self.compute_param_map)
        self.param_map_draw_btn.clicked.connect(self.draw_param_map)

    def setup_layout(self):
        self.setLayout(
            hStack(
                vStack(
                    hStack(self.param_wgt),
                    hStack(self.param_map_compute_btn, self.param_map_draw_btn),
                    hStack(self.random_seq_gen_btn, self.random_seq_reset_btn),
                    self.root_seq_edit
                ),
                vStack(
                    # self.phase_wgt,
                    self.basins_wgt,
                    hStack(self.refresh_btn, self.clear_cb, self.period_label),
                    self.alpha_slider_wgt,
                    self.h_slider_wgt,
                )
            )
        )
        self.layout().setSpacing(0)

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
        with self.compute_lock:
            self.param_wgt.setImage(self.param_placeholder())

    def compute_and_draw_basins(self, *_):
        h, alpha = self.param_wgt.value()

        try:
            seq = self.parse_root_sequence()
        except:
            seq = None

        with self.compute_lock:
            self.basins.compute_points(
                alpha=alpha,
                h=h,
                c=cfg.C,
                skip=cfg.basins_skip,
                root_seq=seq
            )
            image = self.basins.draw_points()

            self.basins_wgt.setImage(image)

    def compute_param_map(self, *_):
        with self.compute_lock:
            print("Start computing parameter map")
            t = time.perf_counter()
            try:
                seq = self.parse_root_sequence()
            except:
                seq = None

            self.param.compute_points(
                z0=cfg.param_map_z0,
                c=cfg.C,
                skip=cfg.param_map_skip,
                iter=cfg.param_map_iter,
                tol=cfg.param_map_tolerance,
                root_seq=seq,
                wait=True
            )
            t = time.perf_counter() - t
        print("Computed parameter map in {:.3f} s".format(t))
        self.draw_param_map()

    def draw_param_map(self, *_):
        if self.param.points is None:
            self.compute_param_map()
        with self.compute_lock:
            image, periods = self.param.display(num_points=cfg.param_map_iter)
            self.param_wgt.setImage(image)
            self.period_map = periods

    def draw_phase(self, *_):
        h, alpha = self.param_wgt.value()

        x_px, y_px = self.param_wgt._imageWidget.targetPx()
        # print(x_px, y_px)
        if self.period_map is not None and 0 <= x_px < cfg.param_map_image_shape[0] and 0 <= y_px < cfg.param_map_image_shape[1]:
            y, x = int((y_px)), int((x_px))
            per = self.period_map[y][x]
            self.period_label.setText(
                "Detected period: {}".format("<chaos({})>".format(per) if per > 0.25 * cfg.param_map_iter else per)
            )

        try:
            seq = self.parse_root_sequence()
        except:
            seq = None

        with self.compute_lock:
            image = self.phase(
                alpha=alpha,
                h=h,
                c=cfg.C,
                grid_size=cfg.phase_grid_size,
                iterCount=cfg.phase_iter,
                skip=cfg.phase_skip,
                root_seq=seq,
                clear_image=self.clear_cb.isChecked(),
                z0=cfg.phase_z0
            )
            self.phase_wgt.setImage(image)


if __name__ == '__main__':
    CourseWork().run()
