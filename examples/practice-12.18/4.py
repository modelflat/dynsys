import numpy as np
import numpy
from math import log
import matplotlib.pyplot as plt

# prepare data constants
SKIP = 0
COUNT = 1500
STEP = 0.01
START_POINT = (1.0, 1.0, 1.0)

# compute laponent constants
EPSILON1 = 0.1
EPSILON2 = 1
PERIOD = 80         # period in points


def lorenc(p: (), step):
    sig, bet, r = 10, 8 / 3, 29
    (x, y, z) = p
    res = (
        p[0] + step * (sig * (y - x)),
        p[1] + step * (- x * z + r * x - y),
        p[2] + step * (x * y - bet * z)
    )
    return res


def ressler(p: (), step):
    a, b, r = 0.15, 0.2, 10.0
    (x, y, z) = p
    res = (
        x + step * (-y - z),
        y + step * (x + a*y),
        z + step * (b + z*(x-r))
    )
    return res


def prepare_data(func, dimension: int = 0):
    t = numpy.empty((COUNT, ), dtype=numpy.float)
    point = START_POINT
    for i in range(SKIP):
        point = func(point, STEP)
    for i in range(COUNT):
        point = func(point, STEP)
        t[i] = point[dimension]
    return t


def computeLyapunov(data):
    N = len(data) - 1
    lyapunov = []

    for idx in range(N//2):
        current_idx = idx
        lap = 0.0
        while current_idx < N:
            fiducial_idx = current_idx + PERIOD//2

            # find fiducial point
            while fiducial_idx < N:
                if abs(data[current_idx] - data[fiducial_idx]) < EPSILON1:
                    break
                fiducial_idx += 1

            # нет совпадений
            if fiducial_idx == N:
                break

            eps1_ = abs(data[fiducial_idx] - data[current_idx])
            time = 0

            while (fiducial_idx < N) and (abs(data[fiducial_idx] - data[current_idx]) <= EPSILON2):
                current_idx += 1
                fiducial_idx += 1
                time += 1

            eps2_ = abs(data[fiducial_idx] - data[current_idx])
            lap += log(eps2_/eps1_) / (current_idx)
        lyapunov.append(lap)

    return lyapunov


def another_lyap():
    with open('data.txt', 'r') as f:
        data = [float(i) for i in f.read().split()]

    N = len(data)
    eps = 0.1
    lyapunovs = [[] for i in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            if np.abs(data[i] - data[j]) < eps:
                for k in range(min(N - i, N - j)):
                    if np.abs(data[i + k] - data[j + k]) == 0:
                        continue
                    lyapunovs[k].append(log(np.abs(data[i + k] - data[j + k])))
    # print(lyapunovs)

    with open('lyapunov.txt', 'w') as f:
        for i in range(len(lyapunovs)):
            if len(lyapunovs[i]):
                # string = str((i, sum(lyapunovs[i]) / len(lyapunovs[i])))
                string = "{}\t{}".format(i, sum(lyapunovs[i]) / len(lyapunovs[i]))
                f.write(string + '\n')


if __name__ == '__main__':
    COUNT = 2000



    # another_lyap()

    lap = computeLyapunov(prepare_data(ressler, dimension=0))
    plt.plot(lap)
    plt.show()
