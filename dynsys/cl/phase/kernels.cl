<%include file="/common/rk4.clh" />
<%namespace name="rk4" file="/common/rk4.clh" />

<%include file="/common/util.clh" />
<%namespace name="util" file="/common/util.clh" />

% if system.is_continuous:
<%call expr="rk4.rk4('point_t', 'PARAMETERS', n_systems=1)">
    <%call expr="util.system_eval(system, 'var', 'parameters')" />
</%call>

% else:
void discrete_step(int, point_t*, PARAMETERS*);
void discrete_step(int i, point_t* var, PARAMETERS* parameters) {
    <%call expr="util.system_eval(system, 'var', 'parameters')" />
}
% endif

<%def name="initialize_time()">
% if system.is_continuous:
    real_t t = t_start;
% endif
</%def>

<%def name="do_step()">
% if system.is_continuous:
    rk4(1, t, t + t_step, &var, &parameters);
    t += t_step;
% else:
    discrete_step(i, &var, &parameters);
% endif
</%def>


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

    ${initialize_time()}

    for (int i = 0; i < n_skip; ++i) {
        ${do_step()}
    }

    const ulong pos = (id_1 * get_global_size(1) + id_2) * n_iter;

    for (int i = 0; i < n_iter; ++i) {
        ${do_step()}
        pt_store(var, pos + i, result);
    }
}

% if varying is not UNDEFINED:

<%def name="initialize_grid()">
    PARAMETERS parameters = *_parameters;
    point_t var;
% for i, variable in enumerate(system.variables):
% if variable in varying:
    var.s${i} = bounds[${2 * i}] +
        get_global_id(${varying[variable]}) *
            (bounds[${2 * i + 1}] - bounds[${2 * i}]) / (get_global_size(${varying[variable]}) - 1);
% else:
    var.s${i} = bounds[${2 * i}];
% endif
% endfor
</%def>

kernel void compute_grid(
    const int n_skip,
    const int n_iter,
% if system.is_continuous:
    const real_t t_start,
    const real_t t_step,
% endif
    const global real_t* bounds,
    const global PARAMETERS* _parameters,
    global real_t* result
) {
    ${initialize_grid()}

    ${initialize_time()}

    for (int i = 0; i < n_skip; ++i) {
        ${do_step()}
    }

    const ulong pos = (
        get_global_id(2) * get_global_size(1) * get_global_size(0)
        + get_global_id(1) * get_global_size(0)
        + get_global_id(0)
    ) * n_iter;

    for (int i = 0; i < n_iter; ++i) {
        ${do_step()}
        pt_store(var, pos + i, result);
    }
}


% if variables_to_draw:

inline void draw_point(
    point_t p,
    const global real_t* bounds,
    float4 color,
    write_only image${len(variables_to_draw)}d_t image
) {
<%
    draw_indices = [
        system.variables.index(var)
        for var in variables_to_draw
    ]
    i = draw_indices[0]
    j = draw_indices[1]
    if len(draw_indices) > 2:
        k = draw_indices[2]
%>
% if len(variables_to_draw) == 2:
    const int2 coord = convert_int2_rtz((real2_t)(
        (p.s${i} - bounds[${2*i}]) / (bounds[${2*i + 1}] - bounds[${2*i}]) * (get_image_width(image) - 0.5),
        (p.s${j} - bounds[${2*j}]) / (bounds[${2*j + 1}] - bounds[${2*j}]) * (get_image_height(image) - 0.5)
    ));
% else:
    const int4 coord = (int4)(convert_int3_rtz((real3_t)(
        (p.s${i} - bounds[${2*i}]) / (bounds[${2*i + 1}] - bounds[${2*i}]) * (get_image_width(image) - 0.5),
        get_image_height(image) - 0.5 - (p.s${j} - bounds[${2*j}]) / (bounds[${2*j + 1}] - bounds[${2*j}]) * (get_image_height(image) - 0.5),
        (p.s${k} - bounds[${2*k}]) / (bounds[${2*k + 1}] - bounds[${2*k}]) * (get_image_depth(image) - 0.5)
    )), 0);
    if (coord.z >= get_image_depth(image)) {
        return;
    }
% endif
    if (any(coord < 0) || coord.x >= get_image_width(image) || coord.y >= get_image_height(image)) {
        return;
    }
    write_imagef(image, coord, color);
}

kernel void draw_phase_plane(
    const int n_skip,
    const int n_iter,
% if system.is_continuous:
    const real_t t_start,
    const real_t t_step,
% endif
    const global real_t* bounds,
    const global real_t* drawing_bounds,
    const global PARAMETERS* _parameters,
    write_only image${len(variables_to_draw)}d_t image
) {
    ${initialize_grid()}

    ${initialize_time()}

    for (int i = 0; i < n_skip; ++i) {
        ${do_step()}
    }

    // TODO support colors passed from host
    const float4 color = (float4)(0.0, 0.0, 0.0, 1.0);
    for (int i = 0; i < n_iter; ++i) {
        ${do_step()}
        draw_point(var, drawing_bounds, color, image);
    }
}

% endif
% endif
