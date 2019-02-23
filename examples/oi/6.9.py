systems = """

typedef float real;
typedef uint2 rand_state;

inline real noise(const real D, __private rand_state* state) {
    return D * rand_norm(state);
}

inline void d_system(
    __private real* _x,
    const real x_r,
    __private real* _y,
    const real y_r,
    __private real* _z,
    const real z_r,
    const real a,
    const real p,
    const real c,
    const real om,
    const real D,
    const real eps,
    __private rand_state* state
) {
    const real x = *(_x);
    const real y = *(_y);
    const real z = *(_z);
    
    *(_x) = -om * y - z + noise(D, state);
    *(_y) = om * x + a * y;
    *(_z) = p + z * (x - c);
}

inline void r_system(
    __private real* _x,
    const real x_d,
    __private real* _y,
    const real y_d,
    __private real* _z,
    const real z_d,
    const real a,
    const real p,
    const real c,
    const real om,
    const real D,
    const real eps,
    __private rand_state* state
) {
    const real x = *(_x);
    const real y = *(_y);
    const real z = *(_z);
    
    *(_x) = -om * y - z + eps * (x_d - x) + noise(D, state);
    *(_y) = om * x + a * y;
    *(_z) = p + z * (x - c);
}

inline void systems_step(
    __private real* _x_d,
    __private real* _x_r,
    __private real* _y_d,
    __private real* _y_r,
    __private real* _z_d,
    __private real* _z_r,
    const real a,
    const real p,
    const real c,
    const real om,
    const real D,
    const real eps,
    __private rand_state* state_d,
    __private rand_state* state_r,
) {
    const real x_d = *(_x_d);
    const real x_r = *(_x_r);
    const real y_d = *(_y_d);
    const real y_r = *(_y_r);
    const real z_d = *(_z_d);
    const real z_r = *(_z_r);
    
    d_system(_x_d, x_r, _y_d, y_r, _z_d, z_r, a, p, c, om, D, eps, state_d);
    r_system(_x_r, x_d, _y_r, y_d, _z_r, z_d, a, p, c, om, D, eps, state_r);
}

kernel void phase_twofold(
    const int skip,
    const int iter,
    const real x_min,
    const real x_max,
    const real y_min,
    const real y_max,
    const real z_min,
    const real z_max,
    const real a,
    const real p,
    const real c,
    const real om,
    const real D,
    const real eps,
    rand_state state_d, 
    rand_state state_r,
    write_only image3d_t image
) {
    real x_d = 
    real y_d = 
    real z_d = 
    real x_r = 
    real y_r = 
    real z_r = 
    
    for (int i = 0; i < skip; ++i) {
        systems_step(&x_d, &x_r, &y_d, &y_r, &z_d, &z_r, a, p, c, om, D, eps, &state_d, &state_r);
    }
    
    for (int i = skip; i < iter; ++i) {
        systems_step(&x_d, &x_r, &y_d, &y_r, &z_d, &z_r, a, p, c, om, D, eps, &state_d, &state_r);
    
        // write to image if valid
    }
    
    
}

"""