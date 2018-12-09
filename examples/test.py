import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy
import numpy as np


def ressler(x, y, z, a, b, r, h=0.01):
    new_x = - y - z
    new_y = x + a * y
    new_z = b + (x - r) * z
    return x + h * new_x, y + h * new_y, z + h * new_z


def doPortret(system=ressler, startPoint=(.1, .1, .1), parameters=(.25, .15, 2.5), evaluateNum=100000):
    xyz = numpy.empty((evaluateNum, 3), dtype=np.float64)
    xyz[0] = startPoint
    for i in range(1, evaluateNum):
        xyz[i] = system(*xyz[i - 1], *parameters)
    return xyz[evaluateNum // 4:].T


def slicer(x, y, z, section: float, axis: str):
    ax = {"x": x, "y": y, "z": z}[axis.lower()]
    condition = (ax[1:] < section) & (ax[:-1] >= section)
    return np.extract(condition, x), np.extract(condition, y), np.extract(condition, z)


def plotPhase():
    import time
    t = time.perf_counter()
    xyz = doPortret()
    print("phase    %.3f" % ((time.perf_counter() - t),), "s")
    t = time.perf_counter()
    slicex, slicey, slicez = slicer(*tuple(xyz), 1.5, "x")
    print("poincare %.3f" % ((time.perf_counter() - t),), "s")

    fig = pyplot.figure()
    ax = fig.gca(projection="3d")
    ax.plot(*tuple(xyz), label="ressler's attractor")
    ax.plot(slicex, slicey, slicez, label="ressler's attractor", color="r")
    pyplot.show()


def plotSlice():
    x, y, z = doPortret()
    slicex, slicey, slicez = slicer(x, y, z)
    pyplot.plot(slicey, slicez, "ro", label="ressler's attractor")
    pyplot.show()


def plotDiagram(a=.25, b=.15):
    fig = pyplot.figure()
    ax = fig.gca(projection="3d")

    rRange = [r for r in numpy.arange(2, 5, 0.05)]
    rDiagramData = []
    for r in rRange:

        x, y, z = doPortret(a, b, r, evaluateNum=10000)
        slicex, slicey, slicez = slicer(x, y, z)
        for y, z in zip(slicey, slicez):
            rDiagramData.append([r, y, z])

    rDiagramData = numpy.transpose(rDiagramData)
    r = rDiagramData[0]
    y = rDiagramData[1]
    z = rDiagramData[2]

    ax.plot(r, y, z, "g.", label="biffurcation diagram")

    ax.legend()
    pyplot.show()


# 4 exercise
def partialDerivative(argNum, f, eps):
    def _f(*args):
        v = f(*args)
        args = list(args)
        args[argNum] += eps
        return (f(*args) - v) / eps

    return _f


def doLyapunovIndex(f):
    numpy.log(f)
    ...


# TODO make partial derivative matrix (матрица чатсных производных) см. скрины в ./DynSys

if __name__ == '__main__':
    plotPhase()
    # plotSlice()
    # plotDiagram()
