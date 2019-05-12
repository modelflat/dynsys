import numpy
from scipy.optimize import minimize_scalar


DELTA = 4.66920_16091_02990_7
L_CRITICAL = 1.40115_51890_92050_6


def lam_k(k):
    if k == 1:
        return 1
    if k == 2:
        return 1.31070264134
    lk1 = lam_k(k - 1)
    lk2 = lam_k(k - 2)
    return (lk1 - lk2 + lk1 * DELTA) / DELTA


def initial_guess(k):
    return numpy.array([lam_k(i) for i in range(1, k)])


def log_map_iterated(l, n, x0):
    x = x0
    for _ in range(n):
        x = 1 - l*x*x
    return abs(x - x0)


def refine_guesses(guesses):
    for k, l0 in enumerate(guesses):
        l = minimize_scalar(log_map_iterated,
                             args=(2**k, 0),
                             bounds=(l0 - 0.02, l0 + 0.02),
                             method="bounded"
                             ).x
        print("k = {}: refined l {} -> {}".format(k + 1, l0, l))
        guesses[k] = l
    return guesses


def compute_lambdas(n=16):
    initial = initial_guess(n)
    lambdas = refine_guesses(initial)
    return list(enumerate(lambdas, 1))


if __name__ == '__main__':
    lambdas = compute_lambdas()

    print("l_cr = {}".format(lambdas[-1][1]))
    print("theoretical diff {:.8f}%".format(abs(lambdas[-1][1] - L_CRITICAL) / L_CRITICAL * 100))
