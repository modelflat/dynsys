import numpy

fn_GeneratorKPR = r"""

#define Fz(z) (8.592*(z) - 22*(z)*(z) + 14.408*(z)*(z)*(z))

real3 userFn(real3, real, real, real);
real3 userFn(real3 v, real h, real g, real eps) {
    #define STEP (real)(1e-3)
    real3 p = (real3)(
        2.0f*h*v.x + v.y - g*v.z, 
        -v.x,
        (v.x - Fz(v.z)) / eps
    );
    return v + STEP*p;
}

#define DYNAMIC_COLOR

"""


def GeneratorKPR(v, h, g, eps):
    x, y, z = v
    return numpy.array((
        2*h*x + y - g*z,
        -x,
        (x - (8.592*z - 22*z**2 + 14.408*z**3)) / eps,
    ), dtype=numpy.float)


fn_Ressler = r"""

real3 userFn(real3, real, real, real);
real3 userFn(real3 v, real a, real b, real r) {
    #define STEP (real)(1e-3)
    real3 p = (real3)(
        -v.z - v.y,
        v.x + a*v.z, 
        b + (v.x - r)*v.y
    );
    return v + STEP*p;
}

#define DYNAMIC_COLOR

"""
