<%include file="/common/integrate.clh" />
<%namespace name="integrate" file="/common/integrate.clh" />
<%include file="/common/util.clh" />
<%namespace name="util" file="/common/util.clh" />

% if system.is_continuous:
<%call expr="integrate.rk4_multi('point_t', 'PARAMETERS', n_systems=1)">
<%call expr="util.system_eval(system, 'var', 'parameters')" />
</%call>
% else:
void discrete_step(int, point_t*, PARAMETERS*);
void discrete_step(int i, point_t* var, PARAMETERS* parameters) {

}
% endif

kernel void capture_phase(
    const int n_skip,
    const int n_iter,
% if system.is_continuous:
    const real_t t_start,
    const real_t t_step,
% endif
    const global real_t* init,
    const global PARAMETERS* _parameters,
    global real_t* result
) {
    const int id_1 = get_global_id(0);
    const int id_2 = get_global_id(1);

    point_t var = pt_load(id_1, init);
    PARAMETERS parameters = _parameters[id_2];

% if system.is_continuous:
    real_t t = t_start;
% endif

    for (int i = 0; i < n_skip; ++i) {
% if system.is_continuous:
        rk4_multi(1, t, t + t_step, &var, &parameters);
        t += t_step;
% else:
        discrete_step(i, &var, &parameters);
% endif
    }

    const ulong pos = (id_1 * get_global_size(1) + id_2) * n_iter;

    for (int i = 0; i < n_iter; ++i) {
% if system.is_continuous:
        rk4_multi(1, t, t + t_step, &var, &parameters);
        t += t_step;
% else:
        discrete_step(i, &var, &parameters);
% endif
        pt_store(var, pos + i, result);
    }
}
