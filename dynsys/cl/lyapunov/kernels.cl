<%include file="/common/integrate.clh" />
<%namespace name="integrate" file="/common/integrate.clh" />
<%include file="/common/util.clh" />
<%namespace name="util" file="/common/util.clh" />

inline real_t pt_norm(point_t* a) {
    real_t norm = pt_length(*a);
% if system.dimensions in {1, 2, 3, 4, 8, 16}:
    *a = *a / norm;
% else:
% for i in range(system.dimensions):
    (*a).s${i} /= norm;
% endfor
% endif
    return norm;
}

// a - b * c
inline point_t pt_sub_mul(point_t a, point_t b, real_t c) {
% if system.dimensions in {1, 2, 3, 4, 8, 16}:
    b *= c;
% else:
% for i in range(system.dimensions):
    b.s${i} *= c;
% endfor
% endif
    return pt_sub(a, b);
}

<%call expr="integrate.rk4_multi('point_t', 'PARAMETERS', n_systems=system.dimensions + 1, name='rk4_multi_variations')">
% if system.dimensions == 1:
    const real_t ${system.variables[0]} = *('var[0]');
% else:
% for i, name in enumerate(system.variables):
    const real_t ${name} = var->s${i};
% endfor
% endif
% for var in map(lambda i: f'var + {i}', range(1, system.dimensions + 1)):
<%call expr="util.system_eval(variations, var, 'parameters')" />
% endfor
<%call expr="util.system_eval(system, 'var', 'parameters')" />
</%call>


private void lyapunov_variations(
    int, real_t, real_t, int,
    point_t[${system.dimensions + 1}],
    PARAMETERS*,
    real_t[${system.dimensions}]
);
private void lyapunov_variations(
    int n_iter, real_t t_start, real_t t_step, int n_integrator_steps,
    point_t system_with_variations[${system.dimensions + 1}],
    PARAMETERS* parameters,
    real_t L[${system.dimensions}]
) {
    real_t gsc  [${system.dimensions}];
    real_t norms[${system.dimensions}];
    real_t S    [${system.dimensions}];

    real_t t = t_start;

    for (int i = 0; i < ${system.dimensions}; ++i) {
        S[i] = 0;
    }

    for (int i = 0; i < n_iter; ++i) {
        rk4_multi_variations(n_integrator_steps, t, t + t_step, system_with_variations, parameters);
        t += t_step;

        #define V (system_with_variations + 1)

        // orthonormalize according to Gram-Schmidt
        for (int j = 0; j < ${system.dimensions}; ++j) {
            for (int k = 0; k < j; ++k) {
                gsc[k] = pt_dot(V[j], V[k]);
            }

            for (int k = 0; k < j; ++k) {
                V[j] = pt_sub_mul(V[j], V[k], gsc[k]);
            }

            norms[j] = pt_norm(V + j);
        }

        // Accumulate sum of log of norms
        for (int j = 0; j < ${system.dimensions}; ++j) {
            S[j] += log(norms[j]);
        }
    }

    for (int i = 0; i < ${system.dimensions}; ++i) {
        // TODO ???
        // L[i] = t_step * S[i] / n_iter;
        L[i] = S[i] / (t - t_start);
    }
}

kernel void single_lyapunov_at_point(
    const int n_iter,
    const real_t t_start,
    const real_t t_step,
    const int n_integrator_steps,
    point_t system,
    PARAMETERS parameters,
    global real_t* variations,
    global real_t* L
) {
    point_t system_with_variations[${system.dimensions + 1}];

    system_with_variations[0] = system;
    for (int i = 1; i < ${system.dimensions + 1}; ++i) {
        system_with_variations[i] = pt_load(i - 1, variations);
    }

    real_t L_private[${system.dimensions}];

    lyapunov_variations(
        n_iter, t_start, t_step, n_integrator_steps,
        system_with_variations, &parameters, L_private
    );

    for (int i = 0; i < ${system.dimensions}; ++i) {
        L[i] = L_private[i];
    }
}
