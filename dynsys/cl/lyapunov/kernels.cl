<%include file="/common/integrate.clh" />
<%namespace name="integrate" file="/common/integrate.clh" />
<%include file="/common/util.clh" />
<%namespace name="util" file="/common/util.clh" />

inline real_t pt_norm(point_t* a) {
    real_t norm = pt_length(*a);
% if system.is_represented_by_cl_type:
    *a /= norm;
% else:
% for i in range(system.dimensions):
    (*a).s${i} /= norm;
% endfor
% endif
    return norm;
}

// a - b * c
inline point_t pt_sub_mul(point_t a, point_t b, real_t c) {
% if system.is_represented_by_cl_type:
    b *= c;
% else:
% for i in range(system.dimensions):
    b.s${i} *= c;
% endfor
% endif
    return pt_sub(a, b);
}

<%def name="lyapunov_step_variations()">
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
</%def>

% if system.is_continuous:
<%call expr="integrate.rk4_multi('point_t', 'PARAMETERS', n_systems=system.dimensions + 1, name='rk4_multi_variations')">
<%call expr="lyapunov_step_variations()" />
</%call>
% else:
void discrete_step_variations(int, point_t[${system.dimensions + 1}], PARAMETERS*);
void discrete_step_variations(int i, point_t var[${system.dimensions + 1}], PARAMETERS* parameters) {
<%call expr="lyapunov_step_variations()" />
}
% endif

private void _lyapunov_variations(
    int,
% if system.is_continuous:
    real_t, real_t, int,
% endif
    point_t[${system.dimensions + 1}],
    PARAMETERS*,
    real_t[${system.dimensions}]
);
private void _lyapunov_variations(
    int n_iter,
% if system.is_continuous:
    real_t t_start, real_t t_step,
    int n_integrator_steps,
% endif
    point_t system_with_variations[${system.dimensions + 1}],
    PARAMETERS* parameters,
    real_t L[${system.dimensions}]
) {
    real_t gsc  [${system.dimensions}];
    real_t norms[${system.dimensions}];
    real_t S    [${system.dimensions}];
% if system.is_continuous:
    real_t t = t_start;
% endif
    #define V (system_with_variations + 1)

    for (int i = 0; i < ${system.dimensions}; ++i) {
        S[i] = 0;
    }

    for (int i = 0; i < n_iter; ++i) {
% if system.is_continuous:
        rk4_multi_variations(n_integrator_steps, t, t + t_step, system_with_variations, parameters);
        t += t_step;
% else:
        discrete_step_variations(i, system_with_variations, parameters);
% endif

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

        // accumulate sum of log of norms
        for (int j = 0; j < ${system.dimensions}; ++j) {
            // TODO different sources suggest different bases for this logarithm
            // some suggest 2 and the others suggest e. It shouldn't really matter,
            // but it would be nice to have a single source of truth here
            S[j] += log2(norms[j]);
        }
    }

    for (int i = 0; i < ${system.dimensions}; ++i) {
% if system.is_continuous:
        L[i] = S[i] / (t - t_start);
% else:
        L[i] = S[i] / n_iter;
% endif
    }
    #undef V
}

kernel void lyapunov_variations(
    const int n_iter,
% if system.is_continuous:
    const real_t t_start,
    const real_t t_step,
    const int n_integrator_steps,
% endif
    const global real_t* init,
    const global PARAMETERS* _parameters,
    global real_t* variations,
    global real_t* L
) {
    const int id = get_global_id(0);
    PARAMETERS parameters = _parameters[id];

    point_t system_with_variations[${system.dimensions + 1}];

    system_with_variations[0] = pt_load(0, init);
    for (int i = 1; i < ${system.dimensions + 1}; ++i) {
        system_with_variations[i] = pt_load(i - 1, variations);
    }

    real_t L_private[${system.dimensions}];

    _lyapunov_variations(
        n_iter,
% if system.is_continuous:
        t_start, t_step, n_integrator_steps,
% endif
        system_with_variations, &parameters, L_private
    );

    L += ${system.dimensions} * id;
    for (int i = 0; i < ${system.dimensions}; ++i) {
        L[i] = L_private[i];
    }
}
