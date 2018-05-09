import numpy as np
import matplotlib.pyplot as pp
from matplotlib.animation import FuncAnimation
from sys import argv


COLOR = "r"


class Grid:
    def __init__(self, xrange, yrange):
        fn = lambda x, y: (x, y)
        self.nodes = np.ndarray((len(xrange)*len(yrange), 2), dtype=np.float64)
        i = 0
        for x in xrange:
            for y in yrange:
                self.nodes[i][0], self.nodes[i][1] = fn(x, y)
                i += 1

    def inplace_map(self, fn):
        for i, node in enumerate(self.nodes):
            self.nodes[i] = fn(*node)


class PointsWithTrajectory:

    def __init__(self, lines, points):
        assert len(lines) == len(points)
        self.lines = lines
        self.points = points

    def update(self, i, x, y):
        self.lines[i].set_xdata(np.append(self.lines[i].get_xdata(), x))
        self.lines[i].set_ydata(np.append(self.lines[i].get_ydata(), y))
        self.points[i].set_xdata(x)
        self.points[i].set_ydata(y)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.lines):
            r = self.lines[self.i]
        elif self.i - len(self.lines) < len(self.points):
            r = self.points[self.i - len(self.lines)]
        else:
            raise StopIteration
        self.i += 1
        return r


def make_series_from_grid(grid: Grid):
    return PointsWithTrajectory( [pp.plot(x, y, COLOR + "-")[0] for x, y in grid.nodes],
                                 [pp.plot(x, y, COLOR + ".")[0] for x, y in grid.nodes] )


def compute_step(step_no, series_pack, grid, fn):
    for i, node in enumerate(grid.nodes):
        series_pack.update(i, *fn(node[0], node[1]))
    grid.inplace_map(fn)
    return series_pack


def animate(grid, f, step_count, xlim=(-12,12), ylim=(-12,12)):
    figure = pp.subplots()[0]
    pp.xlim(xlim[0], xlim[1])
    pp.ylim(ylim[0], ylim[1])
    series_pack = make_series_from_grid(grid)
    anim = FuncAnimation(figure, func=compute_step, init_func=lambda: series_pack, fargs=(series_pack, grid, f), frames=step_count, repeat=True, interval=1000/30)
    pp.grid()
    pp.show()


def make_evolving_system(f, g, step):
    return lambda x, y: (x + step*f(x, y), y + step*g(x, y))


# noinspection NonAsciiCharacters
def task_1г(*argv):
    lam = 1
    k = .5
    h = 0.001
    N = 10000

    f = lambda x, y: y
    g = lambda x, y: (lam + k*x**2 - x**4)*y - x

    grid = Grid(np.arange(-10, 10, 1), np.arange(-10, 10, 1))
    animate(grid, make_evolving_system(f, g, h), N)


# noinspection NonAsciiCharacters
def task_1в(*argv):
    lam = .5
    h = 0.01
    N = 10000

    f = lambda x, y: y
    g = lambda x, y: (lam + y**2)*y - x

    grid = Grid(np.arange(-10, 10, 1), np.arange(-10, 10, 1))
    animate(grid, make_evolving_system(f, g, h), N)


tasks = {
    "в": task_1в,
    "г": task_1г
}


if __name__ == '__main__' :
    if len(argv) > 1:
        tasks[argv[1]](*argv[2:])
