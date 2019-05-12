import numpy as np

def diff(fn,
         x,
         params,
         dx):
    return (fn(x + dx, *params) - fn(x, *params)) / dx

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
        x.append(fn(x[i], *params))
    lyap_sum /= (nsum-delta)
    return lyap_sum