import numpy
import matplotlib.pyplot as plt
import matplotlib.widgets as wgt

import numpy as np

def diff(fn,
         x,
         params,
         dx):
    return (fn(x + dx, params) - fn(x, params)) / dx

def logistic_map(x, _lambda):
    x_new = 1 - _lambda * x ** 2
    return x_new

def circular_map(x, Omega, K):
    equation = x + Omega +( (K / (2*numpy.pi) * numpy.sin(x%(2*numpy.pi))) ) #RIGHT VERTION
    return equation

def iterate(_lambda,
            fn,
            k,
            delta,
            x0):
    # iterate function for iterating 1 dimension map
    x_array = [fn(x0, _lambda)]
    for i in range(k):      #k+1, or not?
        x_array.append(fn(x_array[i],_lambda))
    if delta==0:
        return x_array
    else:
        return x_array[-delta:]


from functools import reduce

def pipeline_each(data, fns):
    result = reduce(lambda a, x: list(map(x, a)),
                    fns,
                    data)
    return result


def lyapunov_index(fn,
                   x0,
                   params,
                   nsum ):
    x = [x0]
    dx = 0.0001     # precision need to be high enough
    lyap_sum = 0
    delta = nsum // 2
    for i in range(nsum):
        if i>delta:
            dfdx = diff(fn, x[i], params, dx)
            lyap_sum += np.log(abs(dfdx))
        x.append(fn(x[i], params))
    lyap_sum /= (nsum-delta)
    return lyap_sum


# TODO
# plot bifurcation tree - NOT
# review point's vicinity (x_0 = 0, _lambda = l_critical) - NOT
# NOTE l_critical can be the point of the first bifurcation

#plot bifurcation tree:
# - iterate for certain _lambda
# - find stationary dots - u don't need that
# - mark on the plot
# -  - will use matplotlib library
# - repeat for all others _lambdas in interval [0;5]



# lets test it!
# TODO test! - OK!
# unit tests - tests of each single element of our program - CHECK
# integration tests - global test of 1 feature from _todo - will be at the end

def allLambda(lmin,
              lmax,
              ld,
              *args
              ):
    diagramList = []
    _lambdaRow = numpy.arange(lmin, lmax, ld)
    for _lambda in _lambdaRow:
        diagramList.append(iterate(_lambda, *args))
    return _lambdaRow, diagramList

def diff(fn,
         x,
         params,
         dx):
    return ( fn(x + dx, params) - fn(x, params) ) / dx


def taskB_plot_iterdiag():
    _lambda = 1.4011
    k = 200
    x0 = 0


    # plot basic figures
    na = numpy.arange(-1, 1.00, 0.02)
    f_array =[logistic_map(x, _lambda) for x in na] # parabola
    x = [x for x in na]
    zero_array = [0 for x in na]
    plt.plot(x,f_array) # parabola
    plt.plot(x,x) #diagonal
    plt.plot(x, zero_array) # x axes
    plt.plot(zero_array, x) # y axes

    x_array = iterate(_lambda, logistic_map, k, 0, x0)
    #print(x_array)


    # should be function
    for n in range(0, len(x_array)-1):
        plt.vlines(x_array[n], x_array[n], x_array[n+1], "b")
        plt.hlines(x_array[n+1], x_array[n], x_array[n+1], "b")


    plt.grid()
    plt.show()
    # plt.clf()

    #-----------------------
    def logistic(x, lam):
        return 1.0 - lam * x ** 2

    def RG(fn, k):
        def make_next(fk):
            return lambda x: fk(fk(x * fk(fk(0)))) / fk(fk(0))

        for _ in range(k):
            fn = make_next(fn)

        return fn


    rgs = [RG(lambda x: logistic(x, 1.401), k) for k in range(3)]

    xs = numpy.linspace(-1, 1)
    ys = lambda f: [f(x) for x in xs]
    plt.plot(
        xs, ys(rgs[0]), "r-",
        xs, ys(rgs[1]), "g-",
        xs, ys(rgs[2]), "b-",
    )
    plt.grid()
    plt.show()


def taskD_doFeig():
    """
    read page 220 in Kuznecov for details
    what TODO you need to find stable super-stable points by map equation
    Feigenbaum's equations needed only for the check, not for calculations.
    :return:
    """
    def findFeigdelta(_lambda0,
                      _lambda1,
                      _lambda2,
                      ):
        delta = (_lambda1 - _lambda0) / (_lambda2 - _lambda1)
        return delta

    def findFeig_lambda0(delta,
                         _lambda1,
                         _lambda2,
                         ):
        # @delta = (_lambda1 - _lambda0)/(_lambda2 - _lambda1)
        _lambda0 = _lambda1 - delta * (_lambda2 - _lambda1)
        return _lambda0

    def findFeig_lambda2(delta,
                         _lambda0,
                         _lambda1,
                         ):
        # @delta = (_lambda1 - _lambda0)/(_lambda2 - _lambda1)
        _lambda2 = (_lambda1 - _lambda0) / delta + _lambda1
        return _lambda2

    # delta = 4.669
    # _lambda1 = 1.40115329085
    # _lambda2 = 1.40115518909
    # _lambda0 = findFeig_lambda0(delta, _lambda1, _lambda2)
    # print("_lambda0", _lambda0)

    _lambda0 = 0
    _lambda1 = 1
    _lambda2 = 1.31070264134
    _iteri = 50

    def simple_iterration(x0, fn, params, iter_number):
        x1 = x0

        for i in range(iter_number):
            x0 = x1
            x1 = x0 - fn(x0) / diff(fn, x0, params, dx=0.01)
        return x1

    delta = findFeigdelta(_lambda0, _lambda1, _lambda2)
    print("delta", delta)

    for i in range(_iteri):
        print("lambda 0: ", _lambda0)
        print("lambda 1: ", _lambda1)
        print("lambda 2: ", _lambda2)
        # try:
        #     delta = findFeigdelta(_lambda0, _lambda1, _lambda2)
        # except:
        #     print("exception i =", i)
        #     break
        #TODO by nutons method or division/2 find super stable cicle

        _lambda0 = _lambda1
        _lambda1 = _lambda2
        _lambda2 = findFeig_lambda2(delta, _lambda0, _lambda1)
        # _lambda2 = simple_iterration(_lambda2, findFeig_lambda2, (delta, _lambda0, _lambda1), iter_number=100
    print("final delta", delta)


if __name__ == '__main__':
    # taskB_plot_iterdiag()
    taskD_doFeig()

