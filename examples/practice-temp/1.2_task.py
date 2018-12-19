import numpy
from matplotlib import pyplot as pp
from mpl_toolkits.mplot3d import Axes3D


STEP = 1e-3


def gen(v, h, g, eps):
    x, y, z = v
    return numpy.array((
        2*h*x + y - g*z,
        -x,
        (x - (8.592*z - 22*z**2 + 14.408*z**3)) / eps,
    ), dtype=numpy.float)


def computeTrajectory(fn, startAt=(0, 1, 0), params=(.07, .85, 0.2), iterCount: int = 100000):
    trajectory = numpy.empty((iterCount, 3), dtype=numpy.float)
    trajectory[0] = startAt
    for i in range(1, iterCount):
        trajectory[i] = trajectory[i - 1] + STEP * fn(trajectory[i - 1], *params)
    return trajectory[iterCount // 4 * 3:].T


def findPoincare(x, y, z, section: float, axis: str):
    ax = {"x": x, "y": y, "z": z}[axis.lower()]
    condition = (ax[1:] < section) & (ax[:-1] >= section)
    return numpy.extract(condition, x), numpy.extract(condition, y), numpy.extract(condition, z)


if __name__ == '__main__':
    trajectory = computeTrajectory(gen)
    poincare = findPoincare(*tuple(trajectory), .5, "x")
    fig = pp.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(*trajectory, "g-")
    ax1.plot(*poincare, "r-")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(*poincare[1:])
    pp.show()
