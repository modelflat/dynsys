import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy
import numpy as np
from numpy import cos, sin


def ressler(x, y, z, A, B, g, step=0.01):
    return A + B * (x * cos(x**2 + y ** 2) - y * sin(x**2 + y**2)),\
           B * (x * sin(x**2 + y ** 2) + y * cos(x**2 + y**2)),\
           0


def computeTrajectory(system=ressler, startPoint=(.1, .1, .1),
                      parameters=(4.5, .2, 0), evaluateNum=20000):
    xyz = numpy.empty((evaluateNum, 3), dtype=np.float64)
    xyz[0] = startPoint
    for i in range(1, evaluateNum):
        xyz[i] = system(*xyz[i - 1], *parameters)
    return xyz[evaluateNum // 5 * 4:].T


def findPoincare(x, y, z, section: float, axis: str):
    ax = {"x": x, "y": y, "z": z}[axis.lower()]
    condition = (ax[1:] < section) & (ax[:-1] >= section)
    return np.extract(condition, x), np.extract(condition, y), np.extract(condition, z)


def plotPhase():
    import time
    t = time.perf_counter()
    xyz = computeTrajectory()
    print("phase    %.3f" % ((time.perf_counter() - t),), "s")
    t = time.perf_counter()
    slicex, slicey, slicez = findPoincare(*tuple(xyz), 0, "x")
    print("poincare %.3f" % ((time.perf_counter() - t),), "s")

    fig = pyplot.figure()
    ax = fig.gca(projection="3d")
    ax.plot(*tuple(xyz), "r.", markerSize=1)
    ax.plot(slicex, slicey, slicez, color="r")
    pyplot.show()


def plotDiagram(h=.092, eps=.2, paramRange=numpy.arange(0.4, 1, 0.002)):
    rDiagramData = []
    for g in paramRange:
        trajectory = computeTrajectory(parameters=(h, eps, g), evaluateNum=12000)
        _, pY, pZ = findPoincare(*trajectory, 0, "x")
        for y, z in zip(pY, pZ):
            rDiagramData.append([g, y])
    rDiagramData = numpy.array(rDiagramData, dtype=numpy.float)

    pyplot.plot(*rDiagramData.T, "b.", markerSize=1.2)
    pyplot.show()


if __name__ == '__main__':
    plotPhase()
    plotDiagram()
