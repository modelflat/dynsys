
from scipy.integrate import ode


def f(t, y, sigma, beta, r):
    return [
        sigma*(y[1] - y[0]),
        -y[0]*y[2] + r*y[0]-y[1],
        y[0]*y[1]-beta*y[2]
    ]


def jac(t, y, sigma, beta, r):
    return [
        [-sigma, sigma, 0],
        [r - y[2], -1, -y[0]],
        [y[1], y[0], -beta]
    ]


r = ode(f, jac).set_integrator('dopri5')

y0, t0 = [1.0, 2.0, 3.0], 0
r.set_initial_value(y0, t0)\
    .set_f_params   (10, 8/3, 28)\
    .set_jac_params (10, 8/3, 28)
t1 = 150
dt = 0.01
while r.successful() and r.t < t1:
    r.stiff
    print(r.t + dt, r.integrate(r.t + dt))

# 149.97000000000858 [-1.07624876 -2.11426196 17.90394026]
# 149.98000000000857 [-1.17950001 -2.20910942 17.45687348]
# 149.99000000000856 [-1.28267383 -2.31898176 17.0250136 ]
# 150.00000000000855 [-1.38723672 -2.44450347 16.60838639]