import numpy
from matplotlib import pyplot as pp


STEP = 1e-3
PART_3 = True
TIME = True

if not TIME or PART_3:
    from mpl_toolkits.mplot3d import Axes3D
    assert Axes3D

def GeneratorKPR(v, h, g, eps):
    x, y, z = v
    return numpy.array((
        2*h*x + y - g*z,
        -x,
        (x - (8.592*z - 22*z**2 + 14.408*z**3)) / eps,
    ), dtype=numpy.float)


def computeTrajectory(fn, startAt=(0, .5, .1), params=(.07, .85, .2), iterCount: int = 100000, skip = True):
    trajectory = numpy.empty((iterCount, 3), dtype=numpy.float)
    trajectory[0] = startAt
    for i in range(1, iterCount):
        trajectory[i] = trajectory[i - 1] + STEP * fn(trajectory[i - 1], *params)
    if skip:
        return trajectory[iterCount // 4 * 3:].T
    return trajectory.T


def findPoincare(x, y, z, section: float, axis: str):
    ax = {"x": x, "y": y, "z": z}[axis.lower()]
    condition = (ax[1:] < section) & (ax[:-1] >= section)
    return numpy.extract(condition, x), numpy.extract(condition, y), numpy.extract(condition, z)


def computeBifDiagram(fn, paramRange, h, eps, poincareSlice: float, iterPerValue=10000):
    return numpy.array(tuple(
        (param, pt[1], pt[2])
        for param in paramRange
        for pt in zip(*findPoincare(*computeTrajectory(
            fn, params=(h, eps, param), iterCount=iterPerValue), poincareSlice, "x"))
    ), dtype=numpy.float).T


if __name__ == '__main__':

    if not PART_3:
        it = 100000
        if TIME:
            trajectory = computeTrajectory(GeneratorKPR, iterCount=it, skip=False)
            pp.plot(numpy.linspace(0, it*STEP, it), trajectory[0], label="x")
            pp.plot(numpy.linspace(0, it*STEP, it), trajectory[1], label="y")
            pp.plot(numpy.linspace(0, it*STEP, it), trajectory[2], label="z")
            pp.legend()
        else:
            trajectory = computeTrajectory(GeneratorKPR, iterCount=it)
            ax = pp.figure().gca(projection="3d")
            poincare = findPoincare(*tuple(trajectory), .5, "x")
            ax.plot(*trajectory, "g-")
            ax.plot(*poincare, "r-")
    else:
        ax = pp.figure().gca(projection="3d")
        points = computeBifDiagram(
            GeneratorKPR,
            paramRange=numpy.arange(0.81, 0.86, 0.001),
            h=0.092,
            eps=0.2,
            poincareSlice=0
        )
        ax.plot(*points, "g.")
    pp.show()
