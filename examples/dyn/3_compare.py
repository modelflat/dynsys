# redo from f(iter) to f(Q) - does not work
# TODO f(Q) for not discrete Qs is DONE, now: build the map, like logistic map.

import numpy
import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.widgets
import matplotlib.widgets as wdt
import sup_task_funcs as stf
import timeit

def circular_map_args(x, Omega, K):
    equation = x + Omega +( (K / (2*numpy.pi) * numpy.sin(x%(2*numpy.pi))) ) #RIGHT VERTION
    return equation

def circular_map_kwargs(x, Omega=0.6066, K=1):
    result_x = x + Omega +( (K / (2*numpy.pi) * numpy.sin(x%(2*numpy.pi))) ) #RIGHT VERTION
    return result_x

params = {
    "args" : [],
    "kwargs": {
        "Omega": 0.6066,
        "K": 1,
    },
    "iter_number": 100,
    "dt": 0.01,
    "time_limits": [0, 100, 0.01],
}
state_d = {
    "x_array": numpy.array( [0.01, ] ),
}

def evaluate(system, state_d, params):
    kwargs = params["kwargs"] # must be list
    buff_array = state_d["x_array"]

    for i in np.arange( *params["time_limits"] ):
        x = circular_map_kwargs(buff_array[-1], **kwargs)
        buff_array = numpy.append(buff_array, x)
    state_d["x_array"] = buff_array
    return state_d, params # state_dictionary must be with computed results

def evaluate_list(system, state_d, params):
    kwargs = params["kwargs"] # must be list
    buff_array = [state_d["x_array"][0], ]

    for i in np.arange( *params["time_limits"] ):
        x = circular_map_kwargs(buff_array[-1], **kwargs)
        buff_array.append(x)
    state_d["x_array"] = numpy.array(buff_array)
    return state_d, params # state_dictionary must be with computed results

# ================time test!====
from copy import deepcopy
def test_numpy_append():
    evaluate(circular_map_kwargs, deepcopy(state_d), deepcopy(params))

def test_list_append():
    evaluate_list(circular_map_kwargs, deepcopy(state_d), deepcopy(params))

def find_regimes(state_d, params):
    ...
def do_map():
    #Omega = 0.60666666
    Omega = 0.6
    K = 1
    x = 0
    iter_number = 100

    def iter_map(x, Omega, K, iter_number):
        x_array = [x]
        for i in range(iter_number):
            x_array.append( circular_map_args(x_array[i], Omega, K) )
        return x_array

    fig, ax = matplotlib.pyplot.subplots()

    x_array = iter_map(x, Omega, K, iter_number)

    #my_plot, = matplotlib.pyplot.plot(x_array[:-1],x_array[1:])
    my_plot, = matplotlib.pyplot.plot(x_array)
    slider_ax1 = matplotlib.pyplot.axes([0.25, 0.1, 0.65, 0.03])
    slider_ax2 = matplotlib.pyplot.axes([0.25, 0.15, 0.65, 0.03])
    slider_Omega = matplotlib.widgets.Slider(slider_ax1, "Omega", 0.1, 5, valinit=Omega) #, valstep=0.1)
    slider_K = matplotlib.widgets.Slider(slider_ax2, "K", 0.1, 5, valinit=K) #, valstep=0.1) Does not exist anymore?


    def update(val):
        Omega = slider_Omega.val
        K = slider_K.val
        x_array = iter_map(x, Omega, K, iter_number)
        my_plot.set_ydata(x_array)
        # my_plot.set_xdata(x_array[:-1])
        # my_plot.set_ydata(x_array[1:])



        fig.canvas.draw_idle()

    slider_Omega.on_changed(update)
    slider_K.on_changed(update)

    matplotlib.pyplot.show()


def do_iterlines_array(x0, Omega, K, skip_n_steps=100, present_n_steps=100):
    x_array = [x0]
    #-----------step1--transition_process
    for i in range(skip_n_steps):
        x_array.append( circular_map(x_array[i], Omega, K) )
    x_array = [x_array[-1]]
    #-----------step2--calculating_process
    for i in range(present_n_steps):
        x_array.append(circular_map(x_array[i], Omega, K))
    return x_array

    #

def do_func():
    Omega = 0.6
    K = 12
    x = 0
    xplotarray = [x for x in numpy.arange(0, 2 * numpy.pi, 0.01)]

    def iter_map(x, Omega, K, xplotarray):
        func_array = []
        for x in xplotarray:
            func_array.append(circular_map(x, Omega, K))
        #normalizing
        func_array = [i - 2 * numpy.pi  if i>(2 * numpy.pi) else i for i in func_array]
        return func_array

    func_array = iter_map(x, Omega, K, xplotarray)
    fig, ax = matplotlib.pyplot.subplots()

    matplotlib.pyplot.xlabel("Qn")
    matplotlib.pyplot.ylabel("Qn+1")

    plot_axes = matplotlib.pyplot.gca()
    plot_axes.set_xlim([0, 2 * numpy.pi])
    plot_axes.set_ylim([0, 2 * numpy.pi])
    #matplotlib.pyplot.axis([0, 2 * numpy.pi, 0, 2 * numpy.pi])

    my_plot, = matplotlib.pyplot.plot(xplotarray, func_array)
    matplotlib.pyplot.grid()


    x_n_array = do_iterlines_array(x, Omega, K)

    my_plot2, = matplotlib.pyplot.plot(x_n_array[:-1],x_n_array[1:])

    #---------------------

    slider_ax1 = matplotlib.pyplot.axes([0.25, 0.1, 0.65, 0.03])
    slider_ax2 = matplotlib.pyplot.axes([0.25, 0.15, 0.65, 0.03])
    slider_Omega = matplotlib.widgets.Slider(slider_ax1, "Omega", 0.1, 5, valinit=Omega)
    slider_K = matplotlib.widgets.Slider(slider_ax2, "K", 0.1, 20, valinit=K)

    def update(val):
        Omega = slider_Omega.val
        K = slider_K.val
        #---func data---------
        func_array = iter_map(x, Omega, K, xplotarray)
        my_plot.set_ydata(func_array)
        #---iter data-----
        x_n_array = do_iterlines_array(x, Omega, K)
        my_plot2.set #TODO PLOT LINES HERE!!!!!!!!!!!!!!!!!!!!!
        fig.canvas.draw_idle()

    slider_Omega.on_changed(update)
    slider_K.on_changed(update)

    matplotlib.pyplot.show()

def plot_lyap_map():
    x0 = 0.1
    nsum = 1000
    Omega, K = 0.6066, 1
    params = (Omega, K)
    # stf.lyapunov_index(stf.logistic_map, x0, params, nsum)
    stf.lyapunov_index(circular_map, x0, params, nsum)

if __name__ == '__main__':
    do_map()
    #do_func()
    # plot_lyap_map()

    # print(timeit.timeit(test_numpy_append, number=1), " ms - numpy append")
    # print(timeit.timeit(test_list_append, number=1), " ms - list append")
