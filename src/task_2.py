import numpy as np
import matplotlib.pyplot as pp

g_map = "cubic"

class CubicMap:

    def __init__(self, lam):
        self.lam = lam

    def __call__(self, x):
        return (self.lam - x**2)*x

    def plot_curves(self, style1, style2):
        x = np.linspace(-self.lam, self.lam, 100)
        pp.plot(x, x, style1, x, self(x), style2)

    def compute_iterations(self, x, iter_count):
        result = np.ndarray((iter_count, 2), dtype=np.float64)
        for i in range(0, iter_count, 2):
            x_next = self(x)
            result[i] = x, x_next
            result[i + 1] = x_next, x_next
            x = x_next
        return result

    def plot_iterations(self, x0, iter_count, style):
        xl, yl = self.compute_iterations(x0, iter_count).T
        pp.plot(xl, yl, style)

    def plot(self, x0, iter_count, style="r-", style1="g-", style2="b-"):
        self.plot_curves(style1, style2)
        self.plot_iterations(x0, iter_count, style)
        pp.xlabel('Xn'); pp.ylabel('Xn+1')
        pp.grid()

class LogisticMap(CubicMap):
    def __call__(self, x):
        return self.lam - x**2

def final_point(x0, N, logistic_map, additional_points):
    for i in range(N - additional_points):
        x0 = logistic_map(x0)

    x0_additional_points = []
    for i in range(additional_points):
        x0 = logistic_map(x0)
        x0_additional_points.append(x0)

    return x0_additional_points

def plot_bifurcation_tree(x0_, samples, points_to_draw, lam_range):
    x0 = np.empty((points_to_draw,), dtype=np.float64)
    for i in range(len(x0)):
        x0[i] = x0_

    map_x0 = [[] for i in range(points_to_draw)]

    for i in lam_range:
        x0 = final_point(x0[points_to_draw - 1], samples, g_fact(i), points_to_draw)

        for j in range(points_to_draw):
            map_x0[j].append(x0[j])

    for i in range(points_to_draw):
        pp.plot(lam_range, map_x0[i], 'r.')
        
    pp.grid()

def make_map_factory(g_map):
    if g_map == "logistic":
        return lambda l: LogisticMap(l)
    elif g_map == "cubic":
        return lambda l: CubicMap(l)

g_fact = make_map_factory(g_map)

if __name__=="__main__":
    lam = 1.3
    iters = 100
    x0 = -.05
    lm = g_fact(lam)
    pp.subplot(1,2,1)
    lm.plot(x0, iters)
    pp.subplot(1,2,2)
    plot_bifurcation_tree(x0, 16, 16, np.arange(0, 4, 0.01))
    pp.show()
    
