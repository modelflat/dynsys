
from dynsys.common import *
from dynsys.phase_portrait import PhasePortrait

system_function_source = """

real2 system(real2 v, real lam, real k) {
    real xp = v.y; 
    real yp = (lam + k*v.x*v.x - v.x*v.x*v.x*v.x)*v.y - v.x;
    return (real2)(xp, yp);
}

// #define DYNAMIC_COLOR

"""


class Task1(SimpleApp):

    def __init__(self):
        super().__init__("Task 1")

        self.w, self.h = 512, 512
        self.bounds = Bounds(-2, 2, -2, 2)

        self.iter_count = 1024
        self.draw_last = 1024

        self.attr = PhasePortrait(self.ctx, self.queue, self.w, self.h, self.bounds, system_function_source)
        self.attr_image = ParametrizedImageWidget(self.bounds)

        self.lam = ObservableValue.makeAndConnect(1.0, self.draw_attr)
        self.k = ObservableValue.makeAndConnect(.3, self.draw_attr)

        self.lam_slider = RealSlider.makeAndConnect(-2, 2, self.lam.value(), connect_to=self.lam.setValue)
        self.k_slider = RealSlider.makeAndConnect(-2, 2, self.k.value(), connect_to=self.k.setValue)

        self.setLayout(
            qt_vstack(
                self.attr_image,
                self.k_slider,
                self.lam_slider
            )
        )

    def draw_attr(self, *args):
        self.attr(self.iter_count, self.k.value(), self.lam.value(), draw_last_points=self.draw_last)
        self.attr_image.set_image(self.attr.image)



if __name__ == '__main__':
    Task1().run()